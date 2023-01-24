import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import os
import torch
from torch.nn import Linear
import torch.nn.functional as F 
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

class GCN(torch.nn.Module):
    def __init__(self, num_features, embedding_size=64):
        # Init parent
        super(GCN, self).__init__()
        torch.manual_seed(42)

        # GCN layers ( for Message Passing )
        self.initial_conv = GCNConv(num_features, embedding_size)
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)

        # Output layer ( for scalar output ... REGRESSION )
        self.out = Linear(embedding_size*2, 1)
        

    def forward(self, x, edge_index, batch_index):
        hidden = F.tanh(self.initial_conv(x, edge_index))
        hidden = F.tanh(self.conv1(hidden, edge_index))
        hidden = F.tanh(self.conv2(hidden, edge_index))
        hidden = F.tanh(self.conv3(hidden, edge_index))
          
        ### ( gmp : global MAX pooling, gap : global AVERAGE pooling )
        hidden = torch.cat([gmp(hidden, batch_index), 
                            gap(hidden, batch_index)], dim=1)

        out = F.sigmoid(self.out(hidden))
        return out, hidden