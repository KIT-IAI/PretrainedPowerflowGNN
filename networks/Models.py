# SPDX-License-Identifier: MIT
import torch
import wandb
import torch.nn as nn
from torch_geometric.nn import GCNConv, TAGConv, MessagePassing
from torch_geometric.utils import degree

class MPN_simplenet(nn.Module):
    """Wrapped Message Passing Network
        - Multiple Conv layers
    """
    def __init__(self, nfeature_dim,  output_dim, hidden_dim, n_gnn_layers, K, dropout_rate):
        super().__init__()
        self.nfeature_dim = nfeature_dim
        
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers
        self.K = K
        self.dropout_rate = dropout_rate
        
        #self.edge_aggr = EdgeAggregation(nfeature_dim, efeature_dim, hidden_dim, hidden_dim)
         
        self.convs = nn.ModuleList()
        #Always add a first TAGConv layer with nfeature_dim inputs, hidden_dim outputs
        self.convs.append(TAGConv(nfeature_dim, hidden_dim, K=K))

        if n_gnn_layers == 1:
            self.convs.append(TAGConv(hidden_dim, output_dim, K=K))
        else:
            self.convs.append(TAGConv(hidden_dim, hidden_dim, K=K))

        for l in range(n_gnn_layers-2):
            self.convs.append(TAGConv(hidden_dim, hidden_dim, K=K))
            
        self.convs.append(TAGConv(hidden_dim, output_dim, K=K))

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index

        
        #x = self.edge_aggr(x, edge_index, edge_features)
       
        for i in range(len(self.convs)-1):
            x = self.convs[i](x=x, edge_index=edge_index)
            x = nn.Dropout(self.dropout_rate, inplace=False)(x)
            x = nn.ReLU()(x)
        
        x = self.convs[-1](x=x, edge_index=edge_index)
        
        return x
        
class GCNNet(torch.nn.Module): 
    def __init__(self, num_node_features, hidden_channels, out_channels):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.conv4(x, edge_index)
        
        return x
    
class TAG_LN(nn.Module):
    """multiple tag convolutional layers, with a fully connected layer at the end
    """
    def __init__(self, nfeature_dim, output_dim, hidden_dim, n_gnn_layers, K, dropout_rate):
        super().__init__()
        self.nfeature_dim = nfeature_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers
        self.K = K
        self.dropout_rate = dropout_rate

         
        self.convs = nn.ModuleList()
        #Always add a first TAGConv layer with nfeature_dim inputs, hidden_dim outputs
        self.convs.append(TAGConv(nfeature_dim, hidden_dim, K=K))

        for l in range(n_gnn_layers-1):
            self.convs.append(TAGConv(hidden_dim, hidden_dim, K=K))


         # Add a fully connected layer at the end
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
       
        for i in range(len(self.convs)-1):
            x = self.convs[i](x=x, edge_index=edge_index)
            x = nn.Dropout(self.dropout_rate, inplace=False)(x)
            x = nn.ReLU()(x)
        
        x = self.convs[-1](x=x, edge_index=edge_index)

        # Pass the output through the fully connected layer
        x = self.fc(x)
        
        return x
class TAG_LN_FC3(nn.Module):
    """multiple tag convolutional layers, with a fully connected layer at the end
    """
    def __init__(self, nfeature_dim, output_dim, hidden_dim, n_gnn_layers, K, dropout_rate):
        super().__init__()
        self.nfeature_dim = nfeature_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers
        self.K = K
        self.dropout_rate = dropout_rate

         
        self.convs = nn.ModuleList()
        #Always add a first TAGConv layer with nfeature_dim inputs, hidden_dim outputs
        self.convs.append(TAGConv(nfeature_dim, hidden_dim, K=K))

        for l in range(n_gnn_layers-1):
            self.convs.append(TAGConv(hidden_dim, hidden_dim, K=K))


         # Add a fully connected layer at the end
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
       
        for i in range(len(self.convs)-1):
            x = self.convs[i](x=x, edge_index=edge_index)
            x = nn.Dropout(self.dropout_rate, inplace=False)(x)
            x = nn.ReLU()(x)
        
        x = self.convs[-1](x=x, edge_index=edge_index)

        # Pass the output through the fully connected layer
        x = self.fc1(x)
        x = nn.Dropout(self.dropout_rate, inplace=False)(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.Dropout(self.dropout_rate, inplace=False)(x)
        x = nn.ReLU()(x)
        x = self.fc4(x)
        x = nn.Dropout(self.dropout_rate, inplace=False)(x)
        x = nn.ReLU()(x)
        x = self.fc4(x)
        x = nn.Dropout(self.dropout_rate, inplace=False)(x)
        x = nn.ReLU()(x)
        x = self.fc5(x)
        return x
        
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import EdgeConv

class Edge_LN_L3(nn.Module):
    """multiple Edge convolutional layers, with a fully connected layer at the end
    additionally uses the inverse of the line resistance as edge weights
    """
    def __init__(self, nfeature_dim, efeature_dim, output_dim, hidden_dim, n_gnn_layers, K, dropout_rate):
        super().__init__()
        self.nfeature_dim = nfeature_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers
        self.K = K
        self.dropout_rate = dropout_rate
        self.efeature_dim = efeature_dim
            
        self.convs = nn.ModuleList()
       
        # MLP for EdgeConv
        mlp = Seq(Linear(2 * nfeature_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))

        # Always add a first EdgeConv layer with nfeature_dim inputs, hidden_dim outputs
        self.convs.append(EdgeConv(mlp))

        for l in range(n_gnn_layers-1):
            mlp = Seq(Linear(2 * hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
            self.convs.append(EdgeConv(mlp, 'mean'))

        # Add a fully connected layer at the end
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        # added inverse of resistance (admittance) as edge weight
        # if data.edge_attr is 0, the edge weight is set to 0
        edge_weight= data.edge_attr.squeeze()
        # check for nan in edge_weight
        if torch.isnan(edge_weight).any():
            print('nan in edge weight')
            # edge_weight[torch.isnan(edge_weight)] = 0

        for i in range(len(self.convs)-1):          
            x = self.convs[i](x=x, edge_index=edge_index)
            x = nn.Dropout(self.dropout_rate, inplace=False)(x)
            x = nn.ReLU()(x)
    
        x = self.convs[-1](x=x, edge_index=edge_index)
        
        # Pass the output through the fully connected layer
        x = self.fc(x)
        return x

class TAG_LN_EHUB(nn.Module):
    """TAG Conv to predict ehubload, 
    using multiple tag convolutional layers, loaded from PF model
    added extra layer to transform the extended feature vector
    """
    def __init__(self, pretrained_model, output_dim):
        super().__init__()
        #self.EHUB_node = EHUB_node
        self.dropout_rate = pretrained_model.dropout_rate
        #self.pretrained_model = pretrained_model
        # Add a extralayer to transform the extended feature vector
        self.featureEmbedding = TAGConv(pretrained_model.nfeature_dim + 1, pretrained_model.nfeature_dim, K=1)

        self.convs = nn.ModuleList(pretrained_model.convs)

         # Add a fully connected layer at the end
        self.fc = nn.Linear(pretrained_model.hidden_dim, output_dim)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
       
        # Transform the extended feature vector
        x = self.featureEmbedding(x, edge_index)
        
        for i in range(len(self.convs)-1):
            x = self.convs[i](x=x, edge_index=edge_index)
            x = nn.Dropout(self.dropout_rate, inplace=False)(x)
            x = nn.ReLU()(x)
        
        x = self.convs[-1](x=x, edge_index=edge_index)
        
        # Pass the output through the fully connected layer
        x = self.fc(x)

        #EHUB_bus_value = x[self.EHUB_node]
        # return EHUB_bus_value
        
        return x
       
    
class TAG_LN_EHUB_NoPretrained(nn.Module):
    """TAG Conv to predict ehubload, 
    using multiple tag convolutional layers, without using pretrained model
    """
    def __init__(self, nfeature_dim, output_dim, hidden_dim, n_gnn_layers, K, dropout_rate):
        super().__init__()
        #self.EHUB_node = EHUB_node
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim
        self.K = K

        self.convs = nn.ModuleList()
        # Always add a first TAGConv layer with nfeature_dim inputs, hidden_dim outputs
        self.convs.append(TAGConv(nfeature_dim, hidden_dim, K=K))

        for l in range(n_gnn_layers-1):
            self.convs.append(TAGConv(hidden_dim, hidden_dim, K=K))

        # Add a fully connected layer at the end
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index

        for i in range(len(self.convs)-1):
            x = self.convs[i](x=x, edge_index=edge_index)
            x = nn.Dropout(self.dropout_rate, inplace=False)(x)
            x = nn.ReLU()(x)

        x = self.convs[-1](x=x, edge_index=edge_index)

        # Pass the output through the fully connected layer
        x = self.fc(x)

        #EHUB_bus_value = x[self.EHUB_node]
        #return EHUB_bus_value
        return x
class TAG_FC3_L3(nn.Module):
    """multiple tag convolutional layers, with a fully connected layer at the end
    additionally uses the inverse of the line resistance as edge weights
    """
    def __init__(self, nfeature_dim,efeature_dim, output_dim, hidden_dim, n_gnn_layers, K, dropout_rate):
        super().__init__()
        self.nfeature_dim = nfeature_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers
        self.K = K
        self.dropout_rate = dropout_rate
        self.efeature_dim = efeature_dim
            
        self.convs = nn.ModuleList()
       
        #Always add a first TAGConv layer with nfeature_dim inputs, hidden_dim outputs
        self.convs.append(TAGConv(nfeature_dim, hidden_dim, K=K))

        for l in range(n_gnn_layers-1):
            self.convs.append(TAGConv(hidden_dim, hidden_dim, K=K))


        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        #added inverse of resitance (admittance) as edge weight
        #if data.edge_attr is 0, the edge weight is set to 0
        edge_weight= data.edge_attr.squeeze()
        #check for nan in edge_weight
        if torch.isnan(edge_weight).any():
            print('nan in edge weight')
            #edge_weight[torch.isnan(edge_weight)] = 0


        for i in range(len(self.convs)-1):          
            x = self.convs[i](x=x, edge_index=edge_index, edge_weight=edge_weight)
            x = nn.Dropout(self.dropout_rate, inplace=False)(x)
            x = nn.ReLU()(x)
    
        x = self.convs[-1](x=x, edge_index=edge_index, edge_weight=edge_weight)
        
        # Pass the output through the fully connected layer
        x = self.fc1(x)
        x = nn.Dropout(self.dropout_rate, inplace=False)(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.Dropout(self.dropout_rate, inplace=False)(x)
        x = nn.ReLU()(x)
        x = self.fc4(x)
        x = nn.Dropout(self.dropout_rate, inplace=False)(x)
        x = nn.ReLU()(x)
        x = self.fc4(x)
        x = nn.Dropout(self.dropout_rate, inplace=False)(x)
        x = nn.ReLU()(x)
        x = self.fc5(x)
        return x



class TAG_LN_L3(nn.Module):
    """multiple tag convolutional layers, with a fully connected layer at the end
    additionally uses the inverse of the line resistance as edge weights
    """
    def __init__(self, nfeature_dim,efeature_dim, output_dim, hidden_dim, n_gnn_layers, K, dropout_rate):
        super().__init__()
        self.nfeature_dim = nfeature_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers
        self.K = K
        self.dropout_rate = dropout_rate
        self.efeature_dim = efeature_dim
            
        self.convs = nn.ModuleList()
       
        #Always add a first TAGConv layer with nfeature_dim inputs, hidden_dim outputs
        self.convs.append(TAGConv(nfeature_dim, hidden_dim, K=K))
        
        for l in range(n_gnn_layers-1):
            self.convs.append(TAGConv(hidden_dim, hidden_dim, K=K))


            # Add a fully connected layer at the end
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x = data.x
        if torch.isnan(x).any():
            print('nan in x')
        edge_index = data.edge_index
        if torch.isnan(edge_index).any():
            print('nan in edge index')
        #added inverse of resitance (admittance) as edge weight
        #if data.edge_attr is 0, the edge weight is set to 0
        edge_weight= data.edge_attr.squeeze()
        
        #check for nan in edge_weight
        if torch.isnan(edge_weight).any():
            print('nan in edge weight')
            #edge_weight[torch.isnan(edge_weight)] = 0


        for i in range(len(self.convs)-1):          
            x = self.convs[i](x=x, edge_index=edge_index, edge_weight=edge_weight)
            if torch.isnan(x).any():
                print(f'nan in output of conv layer {i}')
            x = nn.Dropout(self.dropout_rate, inplace=False)(x)
            x = nn.ReLU()(x)
            if torch.isnan(x).any():
                print(f'nan in output after ReLU in layer {i}')
    
        x = self.convs[-1](x=x, edge_index=edge_index, edge_weight=edge_weight)
        if torch.isnan(x).any():
            print('nan in output of last conv layer')
        
        # Pass the output through the fully connected layer
        x = self.fc(x)
        if torch.isnan(x).any():
            print('nan in output of fully connected layer')
        return x
    
class EdgeAggregation(MessagePassing):
    """MessagePassing for aggregating edge features

    """
    def __init__(self, nfeature_dim, efeature_dim, hidden_dim, output_dim):
        super().__init__(aggr='add')
        self.nfeature_dim = nfeature_dim
        self.efeature_dim = efeature_dim
        self.output_dim = output_dim

        # self.linear = nn.Linear(nfeature_dim, output_dim) 
        self.edge_aggr = nn.Sequential(
            nn.Linear(nfeature_dim*2 + efeature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def message(self, x_i, x_j, edge_attr):
        '''
        x_j:        shape (N, nfeature_dim,)
        edge_attr:  shape (N, efeature_dim,)
        '''
        return self.edge_aggr(torch.cat([x_i, x_j, edge_attr], dim=-1)) # PNAConv style
    
    def forward(self, x, edge_index, edge_attr):
        '''
        input:
            x:          shape (N, num_nodes, nfeature_dim,)
            edge_attr:  shape (N, num_edges, efeature_dim,)
            
        output:
            out:        shape (N, num_nodes, output_dim,)
        '''
        
        
        # Step 2: Calculate the degree of each node.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col] 
        
    
        # Step 4: Propagation
        out = self.propagate(x=x, edge_index=edge_index, edge_attr=edge_attr, norm=norm)
        #   no bias here
        
        return out

class TAG_stagy_L3(nn.Module):
    """multiple tag convolutional layers, with a fully connected layer at the end
    additionally uses the inverse of the line resistance as edge weights
    """
    def __init__(self, nfeature_dim,efeature_dim, output_dim, hidden_dim, n_gnn_layers, K, dropout_rate):
        super().__init__()
        self.nfeature_dim = nfeature_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers
        self.K = K
        self.dropout_rate = dropout_rate
        self.efeature_dim = efeature_dim
            
        self.convs = nn.ModuleList()
       
        #Always add a first TAGConv layer with nfeature_dim inputs, hidden_dim outputs
        self.convs.append(TAGConv(nfeature_dim, 32, K=K))
        self.convs.append(TAGConv(32, 64, K=K))
        self.convs.append(TAGConv(64, 128, K=K))
        for l in range(n_gnn_layers-3):
            self.convs.append(TAGConv(hidden_dim, hidden_dim, K=K))

        self.convs.append(TAGConv(hidden_dim, output_dim, K=K))
         

    def forward(self, data):
        x = data.x
        if torch.isnan(x).any():
            print('nan in x')
        edge_index = data.edge_index
        if torch.isnan(edge_index).any():
            print('nan in edge index')
        #added inverse of resitance (admittance) as edge weight
        #if data.edge_attr is 0, the edge weight is set to 0
        edge_weight= data.edge_attr.squeeze()
        
        #check for nan in edge_weight
        if torch.isnan(edge_weight).any():
            print('nan in edge weight')
            #edge_weight[torch.isnan(edge_weight)] = 0


        for i in range(len(self.convs)-1):          
            x = self.convs[i](x=x, edge_index=edge_index, edge_weight=edge_weight)
            if torch.isnan(x).any():
                print(f'nan in output of conv layer {i}')
            x = nn.Dropout(self.dropout_rate, inplace=False)(x)
            x = nn.ReLU()(x)
            if torch.isnan(x).any():
                print(f'nan in output after ReLU in layer {i}')
    
        x = self.convs[-1](x=x, edge_index=edge_index, edge_weight=edge_weight)
        if torch.isnan(x).any():
            print('nan in output of last conv layer')
        
        # Pass the output through the fully connected layer
        x = self.fc(x)
        if torch.isnan(x).any():
            print('nan in output of fully connected layer')
        return x

class TAG_nofc_L3(nn.Module):
    """multiple tag convolutional layers, with a fully connected layer at the end
    additionally uses the inverse of the line resistance as edge weights
    """
    def __init__(self, nfeature_dim,first_layer_dim,efeature_dim, output_dim, hidden_dim, n_gnn_layers, K, dropout_rate):
        super().__init__()
        self.nfeature_dim = nfeature_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers
        self.K = K
        self.dropout_rate = dropout_rate
        self.efeature_dim = efeature_dim
        self.first_layer_dim=first_layer_dim    
        self.convs = nn.ModuleList()
       
        #Always add a first TAGConv layer with nfeature_dim inputs, hidden_dim outputs
        self.convs.append(TAGConv(nfeature_dim, first_layer_dim, K=K))
        self.convs.append(TAGConv(first_layer_dim, hidden_dim, K=K))
        for l in range(n_gnn_layers-3):
            self.convs.append(TAGConv(hidden_dim, hidden_dim, K=K))
        #last layer   
        self.convs.append(TAGConv(hidden_dim, output_dim, K=K))


        

    def forward(self, data):
        x = data.x
        
        edge_index = data.edge_index
       
        #added inverse of resitance (admittance) as edge weight
        #if data.edge_attr is 0, the edge weight is set to 0
        edge_weight= data.edge_attr.squeeze()
        
        


        for i in range(len(self.convs)-1):          
            x = self.convs[i](x=x, edge_index=edge_index, edge_weight=edge_weight)
           
            x = nn.Dropout(self.dropout_rate, inplace=False)(x)
            x = nn.ReLU()(x)
            
    
        x = self.convs[-1](x=x, edge_index=edge_index, edge_weight=edge_weight)
      
        
        
        return x

class TAG_nofc_L2(nn.Module):
    """multiple tag convolutional layers, with a fully connected layer at the end
    additionally uses the inverse of the line resistance as edge weights
    """
    def __init__(self, nfeature_dim, output_dim, hidden_dim, n_gnn_layers, K, dropout_rate):
        super().__init__()
        self.nfeature_dim = nfeature_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_gnn_layers = n_gnn_layers
        self.K = K
        self.dropout_rate = dropout_rate    
        self.convs = nn.ModuleList()
       
        #Always add a first TAGConv layer with nfeature_dim inputs, hidden_dim outputs
        
        self.convs.append(TAGConv(nfeature_dim, hidden_dim, K=K))
        for l in range(n_gnn_layers-2):
            self.convs.append(TAGConv(hidden_dim, hidden_dim, K=K))
        #last layer   
        self.convs.append(TAGConv(hidden_dim, output_dim, K=K))


        

    def forward(self, data):
        x = data.x
        
        edge_index = data.edge_index


        for i in range(len(self.convs)-1):          
            x = self.convs[i](x=x, edge_index=edge_index)
           
            x = nn.Dropout(self.dropout_rate, inplace=False)(x)
            x = nn.ReLU()(x)
            
    
        x = self.convs[-1](x=x, edge_index=edge_index)
      
        
        
        return x
