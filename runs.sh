#!/bin/bash
# SPDX-License-Identifier: MIT

source ".venv/bin/activate"

python train_coldstart.py --K 4 --n_gnn_layers 5 --hidden_dim 128 --grid_split 0
python train_coldstart.py --K 4 --n_gnn_layers 5 --hidden_dim 128 --grid_split 1
python train_coldstart.py --K 4 --n_gnn_layers 5 --hidden_dim 128 --grid_split 2
python trainDER.py --K 4 --n_gnn_layers 4 --hidden_dim 128 --grid_split 0
python trainDER.py --K 4 --n_gnn_layers 4 --hidden_dim 128 --grid_split 1
python trainDER.py --K 4 --n_gnn_layers 4 --hidden_dim 128 --grid_split 2

#python train_pf_only.py --K 4 --n_gnn_layers 4 --hidden_dim 128 --grid_split 1 --num-epochs 100
#python train_pf_only.py --K 4 --n_gnn_layers 4 --hidden_dim 128 --grid_split 2 --num-epochs 100
#python train_pf_only.py --K 4 --n_gnn_layers 4 --hidden_dim 128 --grid_split 0 --num-epochs 100

#python trainDER.py --K 4 --n_gnn_layers 6 --hidden_dim 128 --grid_split 2
#python trainDER.py --K 4 --n_gnn_layers 5 --hidden_dim 128 --grid_split 2
#python trainDER.py --K 4 --n_gnn_layers 4 --hidden_dim 128 --grid_split 2
#python trainDER.py --K 4 --n_gnn_layers 7 --hidden_dim 128 --grid_split 2

#python train_coldstart.py --K 4 --n_gnn_layers 6 --hidden_dim 128 --grid_split 2
#python train_coldstart.py --K 4 --n_gnn_layers 5 --hidden_dim 128 --grid_split 0
#python train_coldstart.py --K 4 --n_gnn_layers 5 --hidden_dim 128 --grid_split 1
#python train_coldstart.py --K 4 --n_gnn_layers 5 --hidden_dim 128 --grid_split 2
#python train_coldstart.py --K 4 --n_gnn_layers 7 --hidden_dim 128 --grid_split 0
#python train_coldstart.py --K 4 --n_gnn_layers 7 --hidden_dim 128 --grid_split 1
#python train_coldstart.py --K 4 --n_gnn_layers 7 --hidden_dim 128 --grid_split 2
#python train_coldstart.py --K 4 --n_gnn_layers 8 --hidden_dim 128 --grid_split 0
#python train_coldstart.py --K 4 --n_gnn_layers 8 --hidden_dim 128 --grid_split 1
#python train_coldstart.py --K 4 --n_gnn_layers 8 --hidden_dim 128 --grid_split 2