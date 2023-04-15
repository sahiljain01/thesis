from torch_geometric.nn import VGAE, GCNConv
import torch.nn.functional as F
import torch

class GraphVAE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(GraphVAE, self).__init__()

        self.input_dim = input_dim

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv_mu = GCNConv(hidden_dim, latent_dim)
        self.conv_logvar = GCNConv(hidden_dim, latent_dim)

    def encode(self, x, edge_index):
        hidden = F.relu(self.conv1(x, edge_index))
        return self.conv_mu(hidden, edge_index), self.conv_logvar(hidden, edge_index)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z, edge_index):
        num_nodes = edge_index.max().item() + 1
        zeros = torch.zeros(num_nodes, self.input_dim, device=z.device)
        hidden = F.relu(self.conv1(zeros, edge_index, z))
        return torch.sigmoid(self.conv_mu(hidden, edge_index))

    def forward(self, x, edge_index):
        print(x)
        print(edge_index)
        mu, logvar = self.encode(x, edge_index)
        z = self.reparameterize(mu, logvar)
        print(z)
        recon_x = self.decode(z, edge_index)

        # Reconstruction loss
        loss = F.binary_cross_entropy(recon_x, x)

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_x, mu, logvar, loss + kl_loss
