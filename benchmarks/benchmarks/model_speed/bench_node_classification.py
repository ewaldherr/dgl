import os
import sys
import dgl
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Process
from dgl.nn import GraphConv
import torch.nn as nn
import torch.nn.functional as F

# Insert the path to the 'benchmarks' directory
sys.path.insert(0, '/dgl/benchmarks/benchmarks')  # Update with the correct path

import utils  # Now this should correctly import the utils module

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

def train(g, features, labels, train_mask, model, epochs=100, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        logits = model(g, features)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def train_partition(part_id, model, graph_name, features, labels, train_mask, train_args):
    part_data = dgl.distributed.load_partition(f'tmp/partitioned/{graph_name}.json', part_id)
    g, nfeat, efeat, partition_book, graph_name, ntypes, etypes = part_data
    train(g, features[g.ndata[dgl.NID]], labels[g.ndata[dgl.NID]], train_mask[g.ndata[dgl.NID]], model, *train_args)

def worker_fn(part_id, model, graph_name, features, labels, train_mask, train_args):
    train_partition(part_id, model, graph_name, features, labels, train_mask, train_args)

@utils.skip_if_gpu()
@utils.benchmark("time", timeout=1200)
@utils.parametrize("graph_name", ["Cora", "Citeseer", "Pubmed"])
@utils.parametrize("vertex_weight", [True, False])
@utils.parametrize("algorithm", ["kahip", "metis", "kahip_fs"])
@utils.parametrize("k", [2, 4, 8])
def track_time(k, algorithm, vertex_weight, graph_name):
    datasets = {
        "Cora": dgl.data.CoraGraphDataset(),
        "Citeseer": dgl.data.CiteseerGraphDataset(),
        "Pubmed": dgl.data.PubmedGraphDataset(),
    }
    graph = datasets[graph_name][0]

    # Get features and labels
    features = graph.ndata['feat']
    labels = graph.ndata['label']
    train_mask = graph.ndata['train_mask']

    # Create model args
    model_args = (graph.ndata['feat'].shape[1], 16, len(torch.unique(labels)))
    train_args = (100, 0.01)

    # Set the number of threads for PyTorch
    num_threads = min(os.cpu_count(), 4)  # Adjust the number based on your needs
    torch.set_num_threads(num_threads)

    with utils.Timer() as t:
        for _ in range(3):
            # Timing
            if algorithm == "kahip_fs":
                dgl.distributed.partition_graph(graph, graph_name, k, "tmp/partitioned", part_method="kahip", balance_edges=vertex_weight, mode=3)
            else:
                dgl.distributed.partition_graph(graph, graph_name, k, "tmp/partitioned", part_method=algorithm, balance_edges=vertex_weight)

            model = GCN(*model_args)
            model.share_memory()  # Allow the model to be shared across processes

            # Use mp.spawn to handle multiprocessing
            def train_partition_fn(part_id):
                worker_fn(part_id, model, graph_name, features, labels, train_mask, train_args)

            mp.spawn(train_partition_fn, nprocs=k, join=True)

    return t.elapsed_secs / 3

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    track_time()
