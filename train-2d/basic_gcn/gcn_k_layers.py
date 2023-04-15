from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
import torch

class GCN(torch.nn.Module):
    """GCN
    dataset.num_node_features = num_features
    dim_h = embedding size
    """
    def __init__(self, num_features=4, dim_h=8, layers=1):
        super(GCN, self).__init__()
        self.layers = layers 

        self.conv_layers = torch.nn.ModuleList()
        for l in range(self.layers):
            if l==0:
                self.conv_layers.append(GCNConv(num_features, dim_h))
            else:
                self.conv_layers.append(GCNConv(dim_h, dim_h))

        self.lin1 = Linear(dim_h*self.layers*3, dim_h*self.layers*3)
        self.lin2 = Linear(dim_h*self.layers*3, 1)

    def forward(self, x, edge_index, batch):
        # Node embeddings 
        hs = {-1: x}
        for l in range(self.layers):
            hs[l] = self.conv_layers[l](hs[l-1], edge_index)
            hs[l] = F.relu(hs[l])

        readouts = ()

        # Graph-level readout
        for index, h in hs.items():
            if index != -1:
                readouts += (global_add_pool(h, batch),)
                readouts += (global_max_pool(h, batch),)
                readouts += (global_mean_pool(h, batch),)
               
        # Concatenate graph embeddings
        h = torch.cat(readouts, dim=1)

        # Classifier
        h = self.lin1(h)
        h = F.relu(h)
        h = self.lin2(h)
        
        return torch.sigmoid(h), h 