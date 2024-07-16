import time
import os
import torch.multiprocessing as mp

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import dgl.data
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv

from .. import utils

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


def train(g, features, labels, train_mask, epochs=100, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        logits = model(g, features)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def train_partition(i, graph_name, features, labels, train_mask):
    part_data = dgl.distributed.load_partition('tmp/partitioned/' + graph_name + '.json', i)
    g, nfeat, efeat, partition_book, graph_name, ntypes, etypes = part_data
    train(g, features[g.ndata[dgl.NID]], labels[g.ndata[dgl.NID]], train_mask[g.ndata[dgl.NID]])
    return True

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

    # Create model
    model = GCN(graph.ndata['feat'].shape[1], 16, len(torch.unique(labels)))
    model.share_memory()

    with utils.Timer() as t:
        for _ in range(3):
            # Timing
            if algorithm == "kahip_fs":
                dgl.distributed.partition_graph(graph, graph_name, k, "tmp/partitioned", part_method="kahip", balance_edges=vertex_weight, mode=3)
            else:
                dgl.distributed.partition_graph(graph, graph_name, k, "tmp/partitioned", part_method=algorithm, balance_edges=vertex_weight)

            # Train model on the partitioned graphs in parallel
            processes = []
            for rank in range(k):
                p = mp.Process(target=train_partition, args=(k, graph_name, features, labels, train_mask))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
    return t.elapsed_secs / 3
