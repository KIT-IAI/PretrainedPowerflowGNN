# SPDX-License-Identifier: MIT
from datetime import datetime
import os
import random

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from datasets.PowerFlowData import PowerFlowData, random_bus_type
from networks.MPN import MPN, MPN_simplenet, SkipMPN, MaskEmbdMPN, MultiConvNet, MultiMPN, MaskEmbdMultiMPN, MaskEmbdMultiMPNComplete
from utils.argument_parser import argument_parser
from utils.training import train_epoch, append_to_json
from utils.evaluation import evaluate_epoch
from utils.custom_loss_functions import Masked_L2_loss, PowerImbalance, MixedMSEPoweImbalance, DER_loss

import wandb


# DER Training (only last layers)

def main():
    # Step 0: Parse Arguments and Setup
    args = argument_parser()
    run_id = datetime.now().strftime("%Y%m%d") + '-' + str(random.randint(0, 9999)) + "DER"
    run_id = 'pretrained' + str(args.hidden_dim) + '-' + str(args.n_gnn_layers) + '-' + str(args.grid_split)
    LOG_DIR = 'logs'
    SAVE_DIR = 'models'
    TRAIN_LOG_PATH = os.path.join(LOG_DIR, 'train_log/train_log_'+run_id+'.pt')
    SAVE_LOG_PATH = os.path.join(LOG_DIR, 'save_logs.json')
    SAVE_MODEL_PATH = os.path.join(SAVE_DIR, 'DER_ONLY_EH_pfnet_grid'+str(args.grid_split)+'.pt')

    # Training parameters
    data_dir = "data"
    nomalize_data = not args.disable_normalize
    num_epochs = args.num_epochs
    loss_fn = DER_loss()
    eval_loss_fn = DER_loss()
    lr = args.lr
    batch_size = args.batch_size
    grid_case = "ehub" # args.case
    conv_K = args.K
    dropout_rate = args.dropout_rate
    der_model = MaskEmbdMultiMPNComplete

    log_to_wandb = args.wandb
    if log_to_wandb:
        wandb.init(project=args.wandb_entity,
                   name=run_id,
                   config=args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(1234)
    np.random.seed(1234)

    grid_splits = [[0, 3, 6], [1, 4, 7], [2, 5, 8]]
    grid_splits = [[0,], [1], [2]]
    grid_split = grid_splits[args.grid_split]
    # Step 1: Load data
    trainset = PowerFlowData(root=data_dir, case=grid_case, split=[.05, .95], task='train', normalize=nomalize_data,
                             transform=random_bus_type, grid_split = grid_split, model='DER', delete_processed = True)
    valset = PowerFlowData(root=data_dir, case=grid_case, split=[.05, .95], task='val', normalize=nomalize_data, grid_split = grid_split, model='DER', delete_processed = True)
    testset = PowerFlowData(root=data_dir, case=grid_case, split=[0.1, 0.8], task='test', normalize=nomalize_data, grid_split = grid_split, model='DER', delete_processed = True)
    
    # save normalizing params
    os.makedirs(os.path.join(data_dir, 'params'), exist_ok=True)
    torch.save({
            'xymean': trainset.xymean,
            'xystd': trainset.xystd,
            'edgemean': trainset.edgemean,
            'edgestd': trainset.edgestd,
        }, os.path.join(data_dir, 'params', f'data_params_{run_id}.pt'))
        
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    ## [Optional] physics-informed loss function
    if args.train_loss_fn == 'power_imbalance':
        # overwrite the loss function
        loss_fn = PowerImbalance(*trainset.get_data_means_stds()).to(device)
    elif args.train_loss_fn == 'masked_l2':
        loss_fn = Masked_L2_loss(regularize=args.regularize, regcoeff=args.regularization_coeff)
    elif args.train_loss_fn == 'mixed_mse_power_imbalance':
        loss_fn = MixedMSEPoweImbalance(*trainset.get_data_means_stds(), alpha=0.9).to(device)
    else:
        loss_fn = torch.nn.MSELoss()
        
    loss_fn = DER_loss()
    # Step 2: Create model and optimizer (and scheduler)
    node_in_dim, node_out_dim, edge_dim = trainset.get_data_dimensions()
    assert node_in_dim == 6

    # create new model for DER optimization

    der_model = der_model(
        nfeature_dim=node_in_dim,
        efeature_dim=edge_dim,
        output_dim=1,
        hidden_dim=args.hidden_dim,
        n_gnn_layers=args.n_gnn_layers,
        K=conv_K,
        dropout_rate=dropout_rate
    ).to(device)

    
    #calculate model size
    pytorch_total_params = sum(p.numel() for p in der_model.parameters())
    print("Total number of parameters: ", pytorch_total_params)
    
    optimizer = torch.optim.AdamW(der_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=num_epochs)

    # Step 3: Train der_model
    best_train_loss = 10000.
    best_val_loss = 10000.
    train_log = {
        'train': {
            'loss': []},
        'val': {
            'loss': []},
    }
    for epoch in range(num_epochs):
        train_loss = train_epoch(
            der_model, train_loader, loss_fn, optimizer, device)
        val_loss = evaluate_epoch(der_model, val_loader, eval_loss_fn, device)
        scheduler.step()
        train_log['train']['loss'].append(train_loss)
        train_log['val']['loss'].append(val_loss)

        if log_to_wandb:
            wandb.log({'train_loss': train_loss,
                      'val_loss': val_loss,
                      'den_val_loss': val_loss*(valset.xystd[-1][-1]+1e-7)})

        if train_loss < best_train_loss:
            best_train_loss = train_loss

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if args.save:
                _to_save = {
                    'epoch': epoch,
                    'args': args,
                    'val_loss': best_val_loss,
                    'model_state_dict': der_model.state_dict(),
                }
                os.makedirs('models', exist_ok=True)
                torch.save(_to_save, SAVE_MODEL_PATH)
                append_to_json(
                    SAVE_LOG_PATH,
                    run_id,
                    {
                        'val_loss': f"{best_val_loss: .4f}",
                        'train_log': TRAIN_LOG_PATH,
                        'saved_file': SAVE_MODEL_PATH,
                        'epoch': epoch,
                        'model': 'DER',#args.model,
                        'train_case': args.case,
                        'train_loss_fn': args.train_loss_fn,
                        'args': vars(args)
                    }
                )
                torch.save(train_log, TRAIN_LOG_PATH)

        print(f"Epoch {epoch+1} / {num_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, best_val_loss={best_val_loss:.4f}")
    print(f"Training Complete. Best validation loss: {best_val_loss:.4f}")
    
    # Step 4: Evaluate model
    if args.save:
        _to_load = torch.load(SAVE_MODEL_PATH)
        der_model.load_state_dict(_to_load['model_state_dict'])
        test_loss = evaluate_epoch(der_model, test_loader, eval_loss_fn, device)
        print(f"Test loss: {best_val_loss:.4f}")
        if log_to_wandb:
            wandb.log({'test_loss': test_loss})
            wandb.log({'den_test_loss': test_loss*(testset.xystd[-1][-1]+1e-7)})

    # Step 5: Save results
    os.makedirs(os.path.join(LOG_DIR, 'train_log'), exist_ok=True)
    if args.save:
        torch.save(train_log, TRAIN_LOG_PATH)


if __name__ == '__main__':
    main()