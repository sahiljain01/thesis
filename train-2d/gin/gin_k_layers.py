from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
import torch

class GIN(torch.nn.Module):
    """GIN
    dataset.num_node_features = num_features
    dim_h = embedding size
    """
    def __init__(self, num_features=2, dim_h=8, layers=1):
        super(GIN, self).__init__()
        self.layers = layers 

        self.conv_layers = torch.nn.ModuleList()
        for l in range(self.layers):
            if l==0:
                self.conv_layers.append(GINConv(
                    Sequential(Linear(num_features, dim_h),
                            BatchNorm1d(dim_h), ReLU(),
                            Linear(dim_h, dim_h), ReLU())))
            else:
                self.conv_layers.append(GINConv(
            Sequential(Linear(dim_h, dim_h), BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU())))

        self.lin1 = Linear(dim_h*self.layers*3, dim_h*self.layers*3)
        self.lin2 = Linear(dim_h*self.layers*3, 1)

    def forward(self, x, edge_index, batch):
        # Node embeddings 
        hs = {-1: x}
        for l in range(self.layers):
            hs[l] = self.conv_layers[l](hs[l-1], edge_index)
            # print(hs[l].shape)

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
        h = h.relu()
        h = self.lin2(h)
        
        return F.sigmoid(h), h, readouts
