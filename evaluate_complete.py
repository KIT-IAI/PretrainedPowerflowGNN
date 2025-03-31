# SPDX-License-Identifier: MIT
from datetime import datetime
import os
import random

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from datasets.PowerFlowData import PowerFlowData, random_bus_type

from networks.MPN import MPN, MPN_simplenet, SkipMPN, MaskEmbdMPN, MultiConvNet, MultiMPN, MaskEmbdMultiMPN, MPN_DER, MaskEmbdMultiMPNComplete
from utils.argument_parser import argument_parser
from utils.training import train_epoch, append_to_json
from utils.custom_loss_functions import Masked_L2_loss, PowerImbalance, MixedMSEPoweImbalance, DER_loss

import wandb

def main():
    # Step 0: Parse Arguments and Setup
    args = argument_parser()
    run_id = datetime.now().strftime("%Y%m%d") + '-' + str(random.randint(0, 9999)) + "pfnet"
    SAVE_DIR = 'models'
    coldstart = args.use_coldstart
    use_power_flow_model = args.use_pf_net
    net_type = 'all'
    der_net_type = net_type
    if args.use_generalizing_net:
        net_type = 'grid' + str(args.grid_split)
        der_net_type = net_type
    der_net_type = 'pfnet_' + net_type
        
    if coldstart:
        SAVE_DER_MODEL_PATH = os.path.join(SAVE_DIR, 'DER_ONLY_EH_coldstart_'+net_type+'.pt')
    else:
        SAVE_DER_MODEL_PATH = os.path.join(SAVE_DIR, 'DER_ONLY_EH_'+der_net_type+'.pt')
    SAVE_PF_MODEL_PATH = os.path.join(SAVE_DIR, 'model_20250325-3868pfnet_g0.pt')

    print(f"Loading DER model from {SAVE_DER_MODEL_PATH}")
    # Training parameters
    data_dir = "data"
    nomalize_data = not args.disable_normalize
    loss_fn = Masked_L2_loss(regularize=args.regularize, regcoeff=args.regularization_coeff)
    batch_size = args.batch_size
    grid_case = "ehub"
    
    dropout_rate = args.dropout_rate
    pfmodel = MaskEmbdMultiMPN
    der_model = MPN_DER

    log_to_wandb = args.wandb
    wandb_entity = args.wandb_entity
    if log_to_wandb:
        wandb.init(project="PowerFlowNetEHPost",
                   name=run_id,
                   config=args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(1234)
    np.random.seed(1234)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    #grid_splits = [[0, 3,], [1, 4], [2, 5,]] # for generalization
    if args.use_generalizing_net:
        grid_splits = [[0, 3,], [1, 4], [2, 5,]]
    else:
        grid_splits = [[0,1,2], [3,4,5], [6,7,8]] # for no generalization
    grid_split = grid_splits[args.grid_split]
    if coldstart:
        testset = PowerFlowData(root=data_dir, case=grid_case, split=[0.1, 0.8], task='test', normalize=nomalize_data, grid_split = grid_split, model='coldstart', delete_processed=True)
    else:
        testset = PowerFlowData(root=data_dir, case=grid_case, split=[0.1, 0.8], task='test', normalize=nomalize_data, grid_split = grid_split, model='DER', delete_processed=True)
    
    pftestset = PowerFlowData(root=data_dir, case=grid_case, split=[0.1, 0.8], task='test', normalize=nomalize_data, grid_split = grid_split, model = 'PF', delete_processed=False)
    
    # save normalizing params
    os.makedirs(os.path.join(data_dir, 'params'), exist_ok=True)
    torch.save({
            'xymean': testset.xymean,
            'xystd': testset.xystd,
            'edgemean': testset.edgemean,
            'edgestd': testset.edgestd,
        }, os.path.join(data_dir, 'params', f'data_params_{run_id}.pt'))
    
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    loss_fn = DER_loss()
    # Step 2: Create model and optimizer (and scheduler)
    node_in_dim, node_out_dim, edge_dim = testset.get_data_dimensions()
    
    if coldstart:
        assert node_in_dim == 4
    else:
        assert node_in_dim == 6

    der_model = MaskEmbdMultiMPNComplete(
        nfeature_dim=node_in_dim,
        efeature_dim=edge_dim,
        output_dim=1,
        hidden_dim=128,
        n_gnn_layers=4,
        K=4,
        dropout_rate=dropout_rate).to(device)

    if coldstart:
        der_model = MaskEmbdMultiMPNComplete(
            nfeature_dim=node_in_dim,
            efeature_dim=edge_dim,
            output_dim=1,
            hidden_dim=256,
            n_gnn_layers=5,
            K=4,
            dropout_rate=dropout_rate
        ).to(device)
    
    der_to_load = torch.load(SAVE_DER_MODEL_PATH, map_location=torch.device('cpu'))
    der_model.load_state_dict(der_to_load['model_state_dict'])

    # Load pre-trained model
    pfmodel = pfmodel(
        nfeature_dim=4,
        efeature_dim=3,
        output_dim=4,
        hidden_dim=128,
        n_gnn_layers=4,
        K=4,
        dropout_rate=dropout_rate
    ).to(device) 
    
    pf_to_load = torch.load(SAVE_PF_MODEL_PATH, map_location=torch.device('cpu'))
    pfmodel.load_state_dict(pf_to_load['model_state_dict'])
    
    #calculate model size
    pytorch_total_params = sum(p.numel() for p in der_model.parameters())
    print("Total number of parameters: ", pytorch_total_params)
    
    # evaluate complete workflow with test set
    pfmodel.eval()
    der_model.eval()
    pbar = tqdm(test_loader, total=len(test_loader), desc='Evaluating DER:')
    num_samples = 0
    total_loss = 0
    pftotal_loss = 0
    pf_loss_fn = torch.nn.MSELoss()
    for data in pbar:
        derdata = data.clone()
        derdata = derdata.to(device)
        pf_in = data.clone()
        pf_in.x = pf_in.x[:,:4]
        pf_in.y = pf_in.y[:,:4]
        pf_in.pred_mask = pftestset.pred_mask[:len(pf_in.pred_mask)]
        pfout = pfmodel(pf_in.to(device))
        pftotal_loss += pf_loss_fn(pfout[:,0], pf_in.y[:,0]).item() * len(data)
        if use_power_flow_model:
            derdata.x[:,:4] = pfout
        out = der_model(derdata)
        loss = loss_fn(out, derdata.y)
        num_samples += len(data)
        total_loss += loss.item() * len(data)

    mean_loss = total_loss / num_samples
    pf_mean_loss = pftotal_loss / num_samples
    print(f"Mean loss: {mean_loss}")
    print(f"Mean loss den: {mean_loss*(testset.xystd[-1][-1]+1e-7)}")
    print(f" pf Mean loss den: {pf_mean_loss*(testset.xystd[-1][0]+1e-7)}")
    # log mean loss to file
    append_to_json(
        SAVE_DER_MODEL_PATH.replace('.pt', '_eval2.json'),
        run_id,
        {
            'mean_loss': f"{mean_loss: .12f}",
            'use_pf_net': str(use_power_flow_model),
            'net_type': net_type,
            'coldstart': str(coldstart),

            'den_test_loss': str(float(mean_loss*(testset.xystd[-1][-1]+1e-7))),
            'den_pf_loss': str(float(pf_mean_loss*(testset.xystd[-1][-1]+1e-7))),
        }
    )


if __name__ == '__main__':
    main()