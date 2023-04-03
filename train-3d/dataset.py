class LamanDataset(InMemoryDataset):
    def __init__(self, root, data_dir, transform=None, pre_transform=None, pre_filter=None):
        self.data_dir = data_dir
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def processed_file_names(self):
        return ['data.pt']
        
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
            x = generate_feature_vector(graph)
            graph_as_data = from_networkx(graph)
            graph_as_data.x = x
            graph_as_data.label = 0
            data_list.append(graph_as_data)
            
        ind = 0
        for graph in total_laman_data[1]:
            ind += 1
            num_nodes = nx.number_of_nodes(graph)
            x = generate_feature_vector(graph)
            graph_as_data = from_networkx(graph)
            graph_as_data.x = x
            graph_as_data.label = 1
            data_list.append(graph_as_data)
            
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])