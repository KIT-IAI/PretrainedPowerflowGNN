[//]: # (SPDX-License-Identifier: MIT)
# Augmented Pre-trained Graph Neural Networks for Grid-Supportive Flexibility Control


![image](https://github.com/KIT-IAI/PretrainedPowerflowGNN/blob/main/architecture.png)
### Introduction

This repository is based on PowerFlowNet, a Graph Neural Network (GNN) architecture for approximating power flows. It is extended by another GNN architecture, inferring the power needed at a specific node in the grid to mitigate congestion. The models are trained on SimBench grids.

The PowerFlowNet Paper can be found at [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0142061524003338)

The PowerFlowNet GitHub Repository can be found at [GitHub](https://github.com/StavrosOrf/PoweFlowNet)

### Description

The datasets can be generated using the `dataset_generator.py` file. In this file, SimBench grids are loaded and the power flow is calculated using the `pandapower` library. An iterative algorithm described in our publication is then used to calculate the power needed at a specified node to mitigate line congestion. The datasets can then be loaded using the `PowerFlowData` class in the `datasets` folder.

The datasets are structured as follows:

#### Edge Features
First two columns contain the adjacency matrix of the grid. The third and fourth columns contain the line resistance and the line impedance, respectively. The fifth column contains a one-hot encoding of the congested line.

```
edge_features[:, 0]: from_bus
edge_features[:, 1]: to_bus
edge_features[:, 2]: line resistance
edge_features[:, 3]: line impedance
edge_features[:, 4]: one-hot encoding of the congested line
```

#### Node Features

The first column contains the name of the bus. The second column contains the type of the bus (slack / gen / load). The third and fourth columns contain the voltage magnitude and the voltage angle, respectively. The fifth and sixth columns contain the active and reactive power, respectively. The seventh column contains the active power needed to mitigate congestion. The eighth column contains a one-hot encoding of the flexibilities node.
```
node_features[:, 0]: Name of the bus
node_features[:, 1]: Type of the bus (slack / gen / load)
node_features[:, 2]: Voltage magnitude
node_features[:, 3]: Voltage angle
node_features[:, 4]: Active power (P) / pu
node_features[:, 5]: Reactive power (Q) / pu
node_features[:, 6]: Active power needed to mitigate congestion (P) / pu
node_features[:, 7]: one-hot encoding of the flexibilities node
```

To train the different models, the files `train_coldstart.py`, `train_pf_only.py` and `trainDER.py` are used. Take a look at `utils/argument_parser.py` to see the different arguments that can be passed to the training scripts.

### Requirements

The code is developed using the following dependencies with Python 3.9:
```
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install simbench
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch_geometric==2.4.0
pip install pyg_lib torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install matplotlib
pip install scikit-cuda scikit-learn plotly seaborn wandb
```

For **CPU** only:
```
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install simbench
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cpu.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.5.0+cpu.html
pip install torch_geometric==2.4.0
pip install pyg_lib torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cpu.html
pip install matplotlib
pip install scikit-learn plotly seaborn wandb
```