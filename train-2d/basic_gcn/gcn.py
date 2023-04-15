import torch
from torch.nn import Linear
import torch.nn.functional as F 
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

class GCN(torch.nn.Module):
    def __init__(self, num_features, embedding_size=4):
        # Init parent
        super(GCN, self).__init__()
        # torch.manual_seed(42)

        # GCN layers ( for Message Passing )
        self.initial_conv = GCNConv(num_features, embedding_size)
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)
        self.conv4 = GCNConv(embedding_size, embedding_size)
        self.conv5 = GCNConv(embedding_size, embedding_size)
        self.conv6 = GCNConv(embedding_size, embedding_size)
        self.conv7 = GCNConv(embedding_size, embedding_size)

        # Output layer ( for scalar output ... REGRESSION )
        self.out = Linear(embedding_size*14, embedding_size*14)
        self.out2 = Linear(embedding_size*14, 1)        

    def forward(self, x, edge_index, batch_index):
        hidden = (self.initial_conv(x, edge_index))
        hidden2 = (self.conv1(hidden, edge_index))
        hidden3 = (self.conv2(hidden2, edge_index))
        hidden4 = (self.conv3(hidden3, edge_index))
        hidden5 = (self.conv4(hidden4, edge_index))
        hidden6 = (self.conv5(hidden5, edge_index))
        hidden7 = (self.conv6(hidden6, edge_index))
          
        ## ( gmp : global MAX pooling, gap : global AVERAGE pooling )
        hidden = torch.cat([gmp(hidden, batch_index), 
                            gap(hidden, batch_index),
                            gmp(hidden2, batch_index), 
                            gap(hidden2, batch_index),
                            gmp(hidden3, batch_index), 
                            gap(hidden3, batch_index),
                            gmp(hidden4, batch_index), 
                            gap(hidden4, batch_index),
                            gmp(hidden5, batch_index), 
                            gap(hidden5, batch_index),
                            gmp(hidden6, batch_index), 
                            gap(hidden6, batch_index),
                            gmp(hidden7, batch_index), 
                            gap(hidden7, batch_index),
                            ], dim=1)
        
        # Classifier
        hidden = self.out(hidden)
        hidden = hidden.relu()
        hidden = self.out2(hidden)
        
        return F.sigmoid(hidden), hidden