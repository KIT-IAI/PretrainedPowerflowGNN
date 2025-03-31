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
from utils.evaluation import evaluate_epoch, evaluate_epoch_v2
from utils.custom_loss_functions import Masked_L2_loss, PowerImbalance, MixedMSEPoweImbalance, DER_loss

import wandb

MODEL_PATH = os.path.join('models', 'model_20250325-3868pfnet_g0.pt')
model = MaskEmbdMultiMPN
testgrid = 0
grid_splits = [[0, 3,], [1, 4], [2, 5,]]
grid_splits = [[0, 1,2], [3,4,5], [6,7,8]]

def main():
    testset = PowerFlowData(root='data', case='ehub', split=[.7, .3], task='test', normalize=True, grid_split = grid_splits[testgrid], model = 'PF', delete_processed=False)
    _to_load = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_loader = DataLoader(testset, batch_size=128, shuffle=False)
    eval_loss_fn = Masked_L2_loss()
    pfmodel = model(
        nfeature_dim=4,
        efeature_dim=3,
        output_dim=4,
        hidden_dim=128,
        n_gnn_layers=4,
        K=4,
        dropout_rate=0.2
    ).to(device)
    pfmodel.load_state_dict(_to_load['model_state_dict'])
    test_loss = evaluate_epoch(pfmodel, test_loader, eval_loss_fn, device)
    print(f"Test loss: {test_loss:.10f}")
    pfmodel.eval()
    pbar = tqdm(test_loader, total=len(test_loader), desc='Evaluating DER:')
    num_samples = 0
    pftotal_loss = 0
    pf_loss_fn = torch.nn.MSELoss()
    for data in pbar:
        out = pfmodel(data.to(device))
        pftotal_loss += pf_loss_fn(out[:,0], data.y[:,0]).item() * len(data)
        num_samples += len(data)
        print(f"denormalized test loss {str(float(pftotal_loss / num_samples *(testset.xystd[-1][0]+1e-7)))}")
    mean_loss = pftotal_loss / num_samples

    print(f"denormalized test loss {str(float(mean_loss*(testset.xystd[-1][0]+1e-7)))}")





if __name__ == '__main__':
    main()