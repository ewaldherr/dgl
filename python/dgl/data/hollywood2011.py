import os
import struct
import dgl
import torch
import numpy as np
from dgl.data import DGLDataset
from dgl.data.utils import download, save_graphs, load_graphs, makedirs

class Hollywood2011Dataset(DGLDataset):
    def __init__(self, 
                 url='http://data.law.di.unimi.it/webdata/hollywood-2011/hollywood-2011.graph',
                 raw_dir=None,
                 force_reload=False,
                 verbose=False,
                 transform=None):
        super(Hollywood2011Dataset, self).__init__(
            name='hollywood2011',
            url=url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform
        )

    def download(self):
        file_path = os.path.join(self.raw_dir, 'hollywood-2011.graph')
        download(self.url, path=file_path)

    def process(self):
        file_path = os.path.join(self.raw_dir, 'hollywood-2011.graph')
        edges = []

        with open(file_path, 'rb') as f:
            # Assuming the first 8 bytes are the number of nodes and edges
            num_nodes = struct.unpack('I', f.read(4))[0]
            num_edges = struct.unpack('I', f.read(4))[0]
            
            for _ in range(num_edges):
                src, dst = struct.unpack('II', f.read(8))
                edges.append((src, dst))

        src_nodes, dst_nodes = zip(*edges)
        self.graph = dgl.graph((src_nodes, dst_nodes), num_nodes=num_nodes)
        
        # Add random node features
        feature_dim = 10
        node_features = torch.tensor(np.random.rand(num_nodes, feature_dim), dtype=torch.float32)
        self.graph.ndata['feat'] = node_features
        
        # Create train, validation, and test masks
        n_nodes = num_nodes
        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train : n_train + n_val] = True
        test_mask[n_train + n_val :] = True
        self.graph.ndata["train_mask"] = train_mask
        self.graph.ndata["val_mask"] = val_mask
        self.graph.ndata["test_mask"] = test_mask
        
        if self.verbose:
            print(f'Graph has {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges')
            print(f'Node features shape: {self.graph.ndata["feat"].shape}')
            print(f'Train mask: {self.graph.ndata["train_mask"].sum()} nodes')
            print(f'Validation mask: {self.graph.ndata["val_mask"].sum()} nodes')
            print(f'Test mask: {self.graph.ndata["test_mask"].sum()} nodes')

    def has_cache(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        return os.path.exists(graph_path)

    def save(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        save_graphs(graph_path, [self.graph])

    def load(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        self.graph = load_graphs(graph_path)[0][0]

    def __getitem__(self, idx):
        assert idx == 0, "This dataset contains only one graph"
        return self.graph

    def __len__(self):
        return 1