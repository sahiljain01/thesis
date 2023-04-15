from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
import torch.nn.functional as F
from torch_geometric.nn import GINConv
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
        self.conv2 = GINConv(
            Sequential(Linear(dim_h, dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv3 = GINConv(
            Sequential(Linear(dim_h, dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv4 = GINConv(
            Sequential(Linear(dim_h, dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv5 = GINConv(
            Sequential(Linear(dim_h, dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv6 = GINConv(
            Sequential(Linear(dim_h, dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv7 = GINConv(
            Sequential(Linear(dim_h, dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        self.conv8 = GINConv(
            Sequential(Linear(dim_h, dim_h), ReLU(),
                       Linear(dim_h, dim_h), ReLU()))
        
        self.lin1 = Linear(dim_h*14, dim_h*14)
        self.lin2 = Linear(dim_h*14, 1)

    def forward(self, x, edge_index, batch):
        # Node embeddings 
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)
        h4 = self.conv4(h3, edge_index)
        h5 = self.conv5(h4, edge_index)
        h6 = self.conv6(h5, edge_index)
        h7 = self.conv7(h6, edge_index)

        # Graph-level readout
        h1_add = global_add_pool(h1, batch)
        h1_max = global_max_pool(h1, batch)
        h2_add = global_add_pool(h2, batch)
        h2_max = global_max_pool(h2, batch)
        h3_add = global_add_pool(h3, batch)
        h3_max = global_max_pool(h3, batch)
        h4_add = global_add_pool(h4, batch)
        h4_max = global_max_pool(h4, batch)
        h5_add = global_add_pool(h5, batch)
        h5_max = global_max_pool(h5, batch)
        h6_add = global_add_pool(h6, batch)
        h6_max = global_max_pool(h6, batch)
        h7_add = global_add_pool(h7, batch)
        h7_max = global_max_pool(h7, batch)

        # Concatenate graph embeddings
        h = torch.concat((h1_add, h2_add, h3_add, h4_add, h5_add, h6_add, h7_add, h1_max, h2_max, h3_max, h4_max, h5_max, h6_max, h7_max), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        # h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)
        
        return F.sigmoid(h), h 
