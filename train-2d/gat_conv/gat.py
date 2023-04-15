import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    def __init__(self, num_features, nhid, first_heads, output_heads, dropout):
        super(GAT, self).__init__()
        self.gc1 = GATConv(nun_features, nhid,
                           heads=first_heads, dropout=dropout)
        self.gc2 = GATConv(nhid*first_heads, dataset.num_classes,
                           heads=output_heads, dropout=dropout)
        self.dropout = dropout

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc2(x, edge_index)
        return F.log_softmax(x, dim=1)
