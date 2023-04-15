import torch
from torch_geometric.data import DataLoader
from VGAE import GraphVAE
import networkx as nx

import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define your training data here as an array of networkx graphs
training_data = [nx.erdos_renyi_graph(100, 0.15) for _ in range(1000)]

def from_networkx_no_feature(graph):
    edge_index = torch.tensor(list(graph.edges)).t().contiguous()
    data = Data(edge_index=edge_index)
    data.num_nodes = graph.number_of_nodes()
    data.graph = graph
    data.x = torch.ones((graph.number_of_nodes(), 1)) # Add a dummy feature tensor
    return data


# Convert each graph to a PyTorch Geometric Data object
data_list = [from_networkx_no_feature(graph) for graph in training_data]

# Create a PyTorch DataLoader for the training data
train_loader = DataLoader(data_list, batch_size=32, shuffle=True)

# Instantiate the VAE model
model = GraphVAE(input_dim=1, hidden_dim=32, latent_dim=16)

# Set the model to device (GPU or CPU)
model = model.to(device)

# Define your optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# loss_fn = model.recon_loss

# Train the model
model.train()
for epoch in range(50):
    for i, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        x, mu, logvar, loss = model(data.x, data.edge_index)
        loss.backward()
        optimizer.step()
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 50, loss.item()))

# Save the trained model
torch.save(model.state_dict(), 'vae_model.pth')
