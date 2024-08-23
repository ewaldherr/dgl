import os
import dgl
import torch
import numpy as np
from .dgl_dataset import DGLBuiltinDataset

class Hollywood2011Dataset(DGLDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=False):
        super(Hollywood2011Dataset, self).__init__(
            name='hollywood2011',
            url=url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose
        )

    def process(self):
        # Path to the edge list file
        file_path = os.path.join(self.raw_dir, 'hollywood2011.edgelist')
        
        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Edge list file not found at {file_path}")

        # Load the edge list from the file
        edges = []
        with open(file_path, 'r') as f:
            for line in f:
                src, dst = map(int, line.strip().split())
                edges.append((src, dst))
        
        # Extract source and destination nodes from the edges
        src_nodes, dst_nodes = zip(*edges)
        num_nodes = max(max(src_nodes), max(dst_nodes)) + 1  # Ensure correct number of nodes
        
        # Create the DGL graph
        self.graph = dgl.graph((src_nodes, dst_nodes), num_nodes=num_nodes)
        
        # Add random node features
        feature_dim = 10
        node_features = torch.tensor(np.random.rand(num_nodes, feature_dim), dtype=torch.float32)
        self.graph.ndata['feat'] = node_features
        
        # Create train, validation, and test masks
        n_train = int(num_nodes * 0.6)
        n_val = int(num_nodes * 0.2)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train : n_train + n_val] = True
        test_mask[n_train + n_val :] = True
        self.graph.ndata["train_mask"] = train_mask
        self.graph.ndata["val_mask"] = val_mask
        self.graph.ndata["test_mask"] = test_mask

    def has_cache(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        return os.path.exists(graph_path)

    def save(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        dgl.data.utils.save_graphs(graph_path, self.graph)

    def load(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        graphs, _ = dgl.data.utils.load_graphs(graph_path)
        self.graph = graphs[0]

    def __getitem__(self, idx):
        return self.graph

    def __len__(self):
        return 1
