# SPDX-License-Identifier: MIT
import os
from typing import Callable, Optional, List, Tuple, Union

import torch
import numpy as np
import torch.utils.data as data
import torch_geometric
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import from_scipy_sparse_matrix, dense_to_sparse
import matplotlib.pyplot as plt

from torch_geometric.datasets import Planetoid

feature_names_from_files = [
    'index',                # starting from 0 
    'type',                 # 
    'voltage magnitude',    # 
    'voltage angle degree', # 
    'Pd',                   # 
    'Qd',                   # 
]

edge_feature_names_from_files = [
    'from_bus',             # 
    'to_bus',               #
    'r pu',                 # 
    'x pu',                 # 
]

def random_bus_type(data: Data) -> Data:
    " data.bus_type -> randomize "
    data.bus_type = torch.randint_like(data.bus_type, low=0, high=2)
    
    return data
    
def denormalize(input, mean, std):
    return input*(std.to(input.device)+1e-7) + mean.to(input.device)

class PowerFlowData(InMemoryDataset):
    """PowerFlowData(InMemoryDataset)

    Parameters:
        root (str, optional) – Root directory where the dataset should be saved. (optional: None)
        pre_filter (callable)- A function 

    Comments:
        we actually do not need adjacency matrix, since we can use edge_index to represent the graph from `edge_features`
        DER input data with 5 node features gets processed to 4 node features (last feature with optimal DER power removed)

    Returns:
        class instance of PowerFlowData
    """
    partial_file_names = [
        "edge_features.npy",
        "node_features.npy",
    ]
    split_order = {
        "train": 0,
        "val": 1,
        "test": 2
    }
    mixed_cases = [
        '118v2',
        '14v2',
    ]
    ehub_cases = [
        'Lineehub1',
        'Lineehub2',
        'Lineehub3',
    ]

    def __init__(self, 
                root: str, 
                case: str = '118', 
                split: Optional[List[float]] = None, 
                task: str = "train", 
                transform: Optional[Callable] = None, 
                pre_transform: Optional[Callable] = None, 
                pre_filter: Optional[Callable] = None,
                normalize=True,
                xymean=None,
                xystd=None,
                edgemean=None,
                edgestd=None,
                grid_split = [0, 3, 6],
                model='DER',# either DER, PF or coldstart
                delete_processed = True):
        
        assert len(split) == 2
        assert task in ["train", "val", "test"]
        assert model in ['DER', 'PF', 'coldstart']
        if model == 'PF':
            #Vm, Va, P, Q
            slack_mask = (0, 0, 0, 0) # 1 = need to predict, 0 = no need to predict
            gen_mask = (1, 1, 0, 0) 
            load_mask = (1, 1, 0, 0)
        elif model == 'DER':
            #Vm, Va, P, Q, DER_P, loc(1-hot)
            slack_mask = (0, 0, 0, 0, 1, 0)
            gen_mask = (0, 0, 0, 0, 1, 0) 
            load_mask = (0, 0, 0, 0, 1, 0)
        elif model == 'coldstart':
            #P, Q, DER_P, loc(1-hot)
            slack_mask = (0, 0, 1, 0)
            gen_mask = (0, 0, 1, 0) 
            load_mask = (0, 0, 1, 0)
        self.bus_type_mask = (slack_mask, gen_mask, load_mask)
        self.normalize = normalize
        self.case = case  # THIS MUST BE EXECUTED BEFORE super().__init__() since it is used in raw_file_names and processed_file_names
        self.split = split
        self.task = task
        self.grid_split = grid_split
        self.model = model
        
        if delete_processed:
            try:
                if task == 'train':
                    os.remove(os.path.join(root,"processed", "case"+f"{self.case}"+f"{self.model}_processed_train.pt"))
                elif task == 'val':
                    os.remove(os.path.join(root,"processed", "case"+f"{self.case}"+f"{self.model}_processed_val.pt"))
                elif task == 'test':
                    os.remove(os.path.join(root,"processed", "case"+f"{self.case}"+f"{self.model}_processed_test.pt"))
            except FileNotFoundError:
                pass  # Do nothing if the file doesn't exist
        super().__init__(root, transform, pre_transform, pre_filter) # self.process runs here
        self.mask = torch.tensor([])
        # assign mean,std if specified
        if xymean is not None and xystd is not None:
            self.xymean, self.xystd = xymean, xystd
            print('xymean, xystd assigned.')
        else:
            self.xymean, self.xystd = None, None
        if edgemean is not None and edgestd is not None:
            self.edgemean, self.edgestd = edgemean, edgestd
            print('edgemean, edgestd assigned.')
        else:
            self.edgemean, self.edgestd = None, None
        self.data, self.slices = self._normalize_dataset(
            *torch.load(self.processed_paths[self.split_order[self.task]]))  # necessary, do not forget!

    def get_data_dimensions(self):
        return self[0].x.shape[1], self[0].y.shape[1], self[0].edge_attr.shape[1]

    def get_data_means_stds(self):
        assert self.normalize == True
        return self.xymean[:1, :], self.xystd[:1, :], self.edgemean[:1, :], self.edgestd[:1, :]

    def _normalize_dataset(self, data, slices, ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.normalize:
            return data, slices

        # normalizing
        # for node attributes
        if self.xymean is None or self.xystd is None:
            xy = data.y
            if self.model == 'DER' or self.model == 'coldstart':
                # remove 1-hot encoding from normalization
                xy = data.y[:,:-1]
            mean = torch.mean(xy, dim=0, keepdim=True)
            std = torch.std(xy, dim=0, keepdim=True)
            self.xymean, self.xystd = mean, std
            # + 0.0000001 to avoid NaN's because of division by zero
        
        if self.model == 'DER' or self.model == 'coldstart':
            data.x[:,:-1] = (data.x[:,:-1] - self.xymean) / (self.xystd + 0.0000001)
            data.y[:,:-1] = (data.y[:,:-1] - self.xymean) / (self.xystd + 0.0000001)
        else:
            data.x = (data.x - self.xymean) / (self.xystd + 0.0000001)
            data.y = (data.y - self.xymean) / (self.xystd + 0.0000001)
        # for edge attributes
        if self.edgemean is None or self.edgestd is None:
            mean = torch.mean(data.edge_attr, dim=0, keepdim=True)
            std = torch.std(data.edge_attr, dim=0, keepdim=True)
            self.edgemean, self.edgestd = mean, std
        data.edge_attr = (data.edge_attr - self.edgemean) / (self.edgestd + 0.0000001)
        return data, slices

    @property
    def raw_file_names(self) -> List[str]:
        if self.case != 'ehub':
            return ["case"+f"{self.case}"+"_"+name for name in self.partial_file_names]
        else:
            return ["case"+f"{case}"+"_"+name for case in self.ehub_cases for name in self.partial_file_names]

    @property
    def processed_file_names(self) -> List[str]:
        return [
            "case"+f"{self.case}"+f"{self.model}_processed_train.pt",
            "case"+f"{self.case}"+f"{self.model}_processed_val.pt",
            "case"+f"{self.case}"+f"{self.model}_processed_test.pt",
        ]

    def len(self):
        return self.slices['x'].shape[0]-1

    def processRawFiles(self, train_val):
        raw_file_test_cases = [j for i in self.grid_split for j in [i*2, (i*2) + 1]]
        print(f"raw file test cases are the following {raw_file_test_cases}")
        part_raw_paths = [self.raw_paths[i] for i in raw_file_test_cases]
        if train_val:
            part_raw_paths = [elem for elem in self.raw_paths if elem not in part_raw_paths]
        assert len(part_raw_paths) % 2 == 0
        raw_paths_per_case = [[self.raw_paths[i], self.raw_paths[i+1],] for i in range(0, len(self.raw_paths), 2)]
        all_case_data = [[],[],[]]
        for case, raw_paths in enumerate(raw_paths_per_case):
            # process multiple cases (if specified) e.g. cases = [14, 118]
            edge_features = torch.from_numpy(np.load(raw_paths[0])).float()
            node_features = torch.from_numpy(np.load(raw_paths[1])).float()

            assert self.split is not None
            if self.split is not None and train_val:
                split_len = [int(len(node_features) * i) for i in self.split]
                # add the remaining to the first split
                split_len[0] += len(node_features) - sum(split_len)

            split_edge_features = [edge_features]
            split_node_features = [node_features]

            # Split necessary
            if train_val:
                split_edge_features = torch.split(edge_features, split_len, dim=0)
                split_node_features = torch.split(node_features, split_len, dim=0)

            for idx, edge_features in enumerate(split_edge_features):
                # shape of element in split_xx: [N, n_edges/n_nodes, n_features]
                # for each case, process train, val, test split
                y = split_node_features[idx][:, :, 2:-2] # shape (N, n_ndoes, 4); Vm, Va, P, Q
                if self.model == 'coldstart':
                    y = split_node_features[idx][:, :, 4:] # shape (N, n_ndoes, 4); P, Q, DER_P, one-hot
                elif self.model == 'DER':
                    y = split_node_features[idx][:, :, 2:] # shape (N, n_ndoes, 4); Vm, Va, P, Q, DER_P, loc(1-hot)
                bus_type = split_node_features[idx][:, :, 1].type(torch.long) # shape (N, n_nodes)
                bus_type_mask = torch.tensor(self.bus_type_mask)[bus_type] # shape (N, n_nodes, 4)
                x = y.clone()*(1.-bus_type_mask) # shape (N, n_nodes, 4)
                e = edge_features # shape (N, n_edges, 4)
                data_list = [
                    Data(
                        x=sample,
                        y=y[i],
                        bus_type=bus_type[i],
                        pred_mask=bus_type_mask[i],
                        edge_index=e[i, :, 0:2].T.to(torch.long),
                        edge_attr=e[i, :, 2:],
                    ) for i, sample in enumerate(x)
                ]

                # only keep data with relevant DER power
                if self.model == 'DER' or self.model == 'coldstart':
                    data_list = [data for data in data_list if sum(data.y[:, 4 if self.model == 'DER' else 2]) > 0 and sum(data.y[:, 4 if self.model == 'DER' else 2]) != 5]

                all_case_data[idx].extend(data_list)

        # Data from all raw files is now in all_case_data
        for idx, case_data in enumerate(all_case_data):
            if len(case_data)> 0:
                data, slices = self.collate(case_data)

                if train_val:
                    # Processed paths 0 and 1 are for train and val
                    torch.save((data, slices), self.processed_paths[idx])
                else:
                    # Processed path 2 is for test
                    torch.save((data, slices), self.processed_paths[2])

    def process(self):
        self.processRawFiles(True)
        self.processRawFiles(False)

def main():
    try:
        # shape = (N, n_edges, 7)       (from, to, ...)
        edge_features = np.load("datasets/raw/case14v2_edge_features.npy")
        # shape = (N, n_nodes, n_nodes)
        adj_matrix = np.load("datasets/raw/case14v2_adjacency_matrix.npy")
        # shape = (N, n_nodes, 9)
        node_features_x = np.load("datasets/raw/case14v2_node_features_x.npy")
        # shape = (N, n_nodes, 8)
        node_features_y = np.load("datasets/raw/case14v2_node_features_y.npy")
    except FileNotFoundError:
        print("File not found.")

    print(f"edge_features.shape = {edge_features.shape}")
    print(f"adj_matrix.shape = {adj_matrix.shape}")
    print(f"node_features_x.shape = {node_features_x.shape}")
    print(f"node_features_y.shape = {node_features_y.shape}")

    trainset = PowerFlowData(root="datasets", case='14v2',
                             split=[.5, .2, .3], task="train")
    train_loader = torch_geometric.loader.DataLoader(
        trainset, batch_size=12, shuffle=True)
    print(len(trainset))
    print(trainset[0])
    print(next(iter(train_loader)))
    pass


if __name__ == "__main__":
    main()