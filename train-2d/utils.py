import gzip
import pickle
import networkx as nx
import pandas as pd
import numpy as np
import random
import os
import torch 

from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_networkx

def clustering_coefficient(G, node):
    ns = [n for n in G.neighbors(node)]
    if len(ns) <= 1:
        return 0
    
    numerator = 0
    denominator = len(ns) * (len(ns) - 1) / 2
    for i in range(0, len(ns)):
        for j in range(i+1, len(ns)):
            n1, n2 = ns[i], ns[j]
            numerator += G.has_edge(n1, n2)
    
    return numerator / denominator
    
def generate_feature_vector(G):
    x = torch.randn(G.number_of_nodes(), 4)
    ind = 0
    for node in G.nodes():
        x[ind][0] = 1 # uniform
        x[ind][1] = G.degree[node] # node degree as a scalar 
        x[ind][2] = clustering_coefficient(G, node) # triangle counting?
        x[ind][3] = ind # node ID features
        ind += 1
    return x