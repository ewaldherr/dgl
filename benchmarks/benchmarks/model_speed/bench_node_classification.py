import time
import os

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


def train(g, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_val_acc = 0
    best_test_acc = 0

    features = g.ndata["feat"]
    labels = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]
    for e in range(100):
        # Forward
        logits = model(g, features)

        # Compute prediction
        pred = logits.argmax(1)

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        # Compute accuracy on training/validation/test
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        

@utils.skip_if_gpu()
@utils.benchmark("time", timeout=1200)
#@utils.parametrize("graph_name", ["cora","pubmed"])
@utils.parametrize("vertex_weight",[True,False])
@utils.parametrize("algorithm", ["kahip","metis"])
@utils.parametrize("k", [2, 4, 8])
#@utils.parametrize("kahip_mode", [1,3])
def track_time(k, algorithm, vertex_weight):
    dataset = dgl.data.CoraGraphDataset()
    graph = dataset[0]
    print(graph.ndata)
    # timing
    with utils.Timer() as t:
        dgl.distributed.partition_graph(graph,"benchcora", k,"tmp/test",part_method = algorithm, balance_edges = vertex_weight)
        for i in range(3):
            for j in range(k):
                part_data = dgl.distributed.load_partition('tmp/test/benchcora.json', j)
                g, nfeat, efeat, partition_book, graph_name, ntypes, etypes = part_data
                if j == 0:
                    print(g.ndata)
                model = GCN(g.ndata["feat"].shape[1], 16, dataset.num_classes)
                train(g, model)
    return t.elapsed_secs / 3

