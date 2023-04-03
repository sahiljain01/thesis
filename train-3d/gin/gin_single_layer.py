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
    def __init__(self, num_features=2, dim_h=2):
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            Sequential(Linear(num_features, dim_h),
                       BatchNorm1d(dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))

        self.lin1 = Linear(dim_h*2, dim_h*2)
        self.lin2 = Linear(dim_h*2, 1)

    def forward(self, x, edge_index, batch):
        # Node embeddings 
        h1 = self.conv1(x, edge_index)

        # Graph-level readout
        h1_add = global_add_pool(h1, batch)
        h1_max = global_max_pool(h1, batch)

        # Concatenate graph embeddings
        h = torch.concat((h1_add, h1_max), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = self.lin2(h)
        
        return F.sigmoid(h), h 
