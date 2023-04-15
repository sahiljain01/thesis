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

class LamanDataset(InMemoryDataset):
    def __init__(self, root, data_dir, transform=None, pre_transform=None, pre_filter=None):
        self.data_dir = data_dir
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def processed_file_names(self):
        return ['data.pt']

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

    def process(self):
        # processing code here
        total_laman_data = None
        with gzip.open(self.data_dir, 'r') as f:
            total_laman_data = pickle.load(f)
            
        data_list = []
        ind = 0
        # convert from graph to Data object
        for graph in total_laman_data[0]:
            ind += 1
            num_nodes = nx.number_of_nodes(graph)
            x = self.generate_feature_vector(graph)
            graph_as_data = from_networkx(graph)
            graph_as_data.x = x
            graph_as_data.label = 0
            data_list.append(graph_as_data)
            
        ind = 0
        for graph in total_laman_data[1]:
            ind += 1
            num_nodes = nx.number_of_nodes(graph)
            x = self.generate_feature_vector(graph)
            graph_as_data = from_networkx(graph)
            graph_as_data.x = x
            graph_as_data.label = 1
            data_list.append(graph_as_data)
            
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])