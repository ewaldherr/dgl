import os
import struct
import dgl
import torch
import numpy as np

class Hollywood2011Dataset(dgl.data.DGLDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=False):
        url = 'http://data.law.di.unimi.it/webdata/hollywood-2011/hollywood-2011.graph'
        super(Hollywood2011Dataset, self).__init__(
            name='hollywood2011',
            url=url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose
        )

    def process(self):
        file_path = os.path.join(self.raw_dir, 'hollywood-2011.graph')
        
        # Validate the properties file
        properties_file = os.path.join(self.raw_dir, 'hollywood-2011.properties')
        if not os.path.exists(properties_file):
            raise FileNotFoundError(f"Properties file not found at {properties_file}")

        # Read the properties file for validation
        with open(properties_file, 'r') as prop_file:
            props = prop_file.readlines()
            node_count = next(line for line in props if line.startswith("nodes=")).split('=')[1].strip()
            edge_count = next(line for line in props if line.startswith("arcs=")).split('=')[1].strip()
            print(f"Nodes (from properties): {node_count}")
            print(f"Edges (from properties): {edge_count}")

        # Read the graph file
        with open(file_path, 'rb') as f:
            # Read and validate header
            header = f.read(8)
            if len(header) != 8:
                raise ValueError(f"Expected to read 8 bytes for header but got {len(header)} bytes")

            num_nodes, num_edges = struct.unpack('II', header)
            print(f'Number of nodes: {num_nodes}')
            print(f'Number of edges: {num_edges}')

            # Read edges in chunks for debugging
            edges = []
            f.seek(8)  # Reset file pointer to just after header
            for i in range(num_edges):
                edge_data = f.read(8)
                if len(edge_data) != 8:
                    print(f"Error at edge {i}: Expected to read 8 bytes but got {len(edge_data)} bytes")
                    break
                src, dst = struct.unpack('II', edge_data)
                edges.append((src, dst))
                if i < 10:  # Print first 10 edges for debugging
                    print(f"Edge {i}: {src} -> {dst}")

        # Validate node and edge numbers
        if len(edges) != num_edges:
            raise ValueError(f"Expected {num_edges} edges but read {len(edges)} edges")

        src_nodes, dst_nodes = zip(*edges)
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
