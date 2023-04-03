import gzip
import pickle
import networkx as nx
import pandas as pd
import numpy as np
import random
import os
import torch
from torch_geometric.data import InMemoryDataset
from torch.utils.data import DataLoader
from torch_geometric.utils import from_networkx, to_networkx

def generate_feature_vector(G):
    x = torch.randn(G.number_of_nodes(), 1)
    ind = 0
    for node in G.nodes():
        x[ind][0] = G.degree[node]
        ind += 1
    return x


