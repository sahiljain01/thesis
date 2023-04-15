import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch_geometric.datasets import ZINC
from torch_geometric.nn import VGAE, GCNConv
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from torch_geometric.data import InMemoryDataset
import gzip
import pickle
from torch_geometric.utils import from_networkx, to_networkx

class LamanDataset(InMemoryDataset):
    def __init__(self, root, data_dir, transform=None, pre_transform=None, pre_filter=None):
        self.data_dir = data_dir
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def generate_feature_vector(self, G):
        x = torch.randn(G.number_of_nodes(), 4)
        ind = 0
        for node in G.nodes():
            x[ind][0] = 1 # uniform
            x[ind][1] = G.degree[node] # node degree as a scalar 
            x[ind][2] = nx.clustering(G, node) # triangle counting?
            x[ind][3] = ind # node ID features
            ind += 1
        return x
    
    def process(self):
        total_laman_data = None
        with gzip.open(self.data_dir, 'r') as f:
            total_laman_data = pickle.load(f)
            
        data_list = []
        for ind, graph in enumerate(total_laman_data[0]):
            x = self.generate_feature_vector(graph)
            graph_as_data = from_networkx(graph)
            graph_as_data.x = x
            graph_as_data.label = 0
            data_list.append(graph_as_data)
            
        for ind, graph in enumerate(total_laman_data[1]):
            x = self.generate_feature_vector(graph)
            graph_as_data = from_networkx(graph)
            graph_as_data.x = x
            graph_as_data.label = 1
            data_list.append(graph_as_data)
            
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

transform = T.Compose([
    T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                      split_labels=True, add_negative_train_samples=True)
])

laman_data = LamanDataset("", "data/new-algo-15-20.pkl.gz", transform=transform)
# laman_data = LamanDataset("", "data/new-algo-15-20.pkl.gz")

from torch.utils.data import random_split

proportions = [.6, .2, .2]
lengths = [int(p * len(laman_data)) for p in proportions]
lengths[-1] = len(laman_data) - sum(lengths[:-1])

generator1 = torch.Generator().manual_seed(42)
train_data_list, test_data_list, val_data_list = random_split(laman_data, lengths, generator=generator1)

print(train_data_list)
print(test_data_list)
# train_data_list, val_data_list, test_data_list = [], [], []

in_channels, out_channels, lr, n_epochs = 4, 16, 1e-2, 2
gen_graphs, threshold, batch_size, add_self_loops = 5, 0.5, 2, False
model = VGAE(VariationalGCNEncoder(in_channels, out_channels))
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data_list, batch_size=batch_size)
test_loader = DataLoader(test_data_list, batch_size=batch_size)

def train():
    model.train()
    loss_all = 0
    for data in train_loader:
        optimizer.zero_grad()
        # print(data)
        data = data[0]
        z = model.encode(data.x, data.edge_index)
        loss = model.recon_loss(z, data.pos_edge_label_index, data.neg_edge_label_index)
        loss = loss + (1 / data.num_nodes) * model.kl_loss()
        loss.backward()
        loss_all += float(loss)
        optimizer.step()
    return loss_all / len(train_loader.dataset)


@torch.no_grad()
def val(loader):
    model.eval()
    auc_all, ap_all = 0, 0

    for data in loader:
        data = data[0]
        z = model.encode(data.x, data.edge_index)
        auc, ap = model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)
        auc_all +=  float(auc)
        ap_all += float(ap)
    return auc_all / len(val_loader.dataset), ap_all / len(val_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()
    graph_adj = []

    for graph, data in enumerate(loader):
        data = data[0]
        z = model.encode(data.x, data.edge_index)
        graph_adj.append(model.decoder.forward_all(z))
        if graph == gen_graphs - 1:
            break
    return graph_adj


for epoch in range(1, n_epochs + 1):
    loss = train()
    auc, ap = val(val_loader)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, AUC: {auc:.4f}, AP: {ap:.4f}')

graphs = np.random.choice(len(test_data_list), gen_graphs, False)

test_graph_list = []
for g_id in graphs:
    test_graph_list.append(test_data_list[g_id])
test_loader = DataLoader(test_graph_list)
recon_adj = test(test_loader)

for graph in range(gen_graphs):
    adj_binary = recon_adj[graph] > 0.75
    print(adj_binary.shape)
    indices = torch.where(adj_binary)
    G = nx.Graph()
    if not add_self_loops:
        edges = [(i, j) for i, j in zip(indices[0].tolist(), indices[1].tolist()) if i != j]
        G.add_edges_from(edges)
    else:
        G.add_edges_from(zip(indices[0].tolist(), indices[1].tolist()))
    nx.draw(G)
    plt.show()