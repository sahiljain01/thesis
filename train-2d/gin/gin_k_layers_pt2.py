from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
import torch.nn as nn
import torch

class GIN(torch.nn.Module):
    def __init__(self, num_features, dim, num_layers):
        super(GIN, self).__init__()

        self.dim = dim
        self.num_features = num_features

        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.fcs = nn.ModuleList()

        self.convs.append(GINConv(nn.Sequential(nn.Linear(num_features, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))))
        self.bns.append(nn.BatchNorm1d(dim))
        self.fcs.append(nn.Linear(num_features, num_features))
        self.fcs.append(nn.Linear(num_features, dim))

        for i in range(self.num_layers-1):
            self.convs.append(GINConv(nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))))
            self.bns.append(nn.BatchNorm1d(dim))
            self.fcs.append(nn.Linear(dim, dim))

        self.lastfc = nn.Linear(dim * (num_layers + 1), 1)
    
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            elif isinstance(m, GINConv):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()

    def forward(self, x, edge_index, batch):
        # x = data.x
        # edge_index = data.edge_index
        # batch = data.batch
        outs = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            outs.append(x)
        
        out = [x]
        for i, x in enumerate(outs):
            x = self.fcs[i](x)
            x = global_add_pool(x, batch)
            # if out is None:
            #     out = x
            # else:
            out = out + [x]

        h = torch.concat(tuple(out), dim=1)
        h = self.lastfc(h)
        return F.sigmoid(h), h