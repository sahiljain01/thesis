import torch
from torch.nn import Sequential, Linear, ReLU, Sigmoid, Dropout

class DeepSets(torch.nn.Module):
    """DeepSets"""
    def __init__(self, num_features=2, hidden_dim=16, num_layers=2):
        super(DeepSets, self).__init__()

        # Define phi function
        self.phi = Sequential(
            Linear(num_features, hidden_dim),
            ReLU(),
            Dropout(0.5)
        )

        # Define rho function
        self.rho = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Dropout(0.5),
            Linear(hidden_dim, 1),
            Sigmoid()
        )

    def forward(self, x, edge_index, batch):
        # Node embeddings 
        node_embeddings = self.phi(x)

        # Aggregate node embeddings to obtain graph embeddings
        graph_embeddings = torch.zeros(batch.max().item() + 1, node_embeddings.shape[1], device=x.device)
        scatter_add(node_embeddings, batch, out=graph_embeddings)

        # Apply rho function to obtain final prediction
        return self.rho(graph_embeddings).view(-1)
