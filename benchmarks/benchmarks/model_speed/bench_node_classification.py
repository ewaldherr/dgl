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


def train(g, features, labels, train_mask, model, epochs=100, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        logits = model(g, features)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
           

@utils.skip_if_gpu()
@utils.benchmark("time", timeout=1200)
@utils.parametrize("graph_name", ["Cora","Citeseer","Pubmed","Amazon Computer","Amazon Photo","PPI"])
@utils.parametrize("vertex_weight",[True,False])
@utils.parametrize("algorithm", ["kahip","metis"])
@utils.parametrize("k", [2, 4, 8])
#@utils.parametrize("kahip_mode", [1,3])
def track_time(k, algorithm, vertex_weight, graph_name):
    datasets = {
    "Cora": CoraGraphDataset(),
    "Citeseer": CiteseerGraphDataset(),
    "Pubmed": PubmedGraphDataset(),
    "Amazon Computer": AmazonCoBuyComputerDataset(),
    "Amazon Photo": AmazonCoBuyPhotoDataset(),
    "Reddit": RedditDataset(),
    "PPI": PPIDataset(),
    }
    graph = datasets[graph_name][0]

    # Get features and labels
    features = graph.ndata['feat']
    labels = graph.ndata['label']
    train_mask = graph.ndata['train_mask']

    # Create model
    model = GCN(graph.ndata['feat'].shape[1], 16, len(torch.unique(labels)))


    # timing
    with utils.Timer() as t:
        dgl.distributed.partition_graph(graph,"benchcora", k,"tmp/test",part_method = algorithm, balance_edges = vertex_weight)
        for i in range(3):
            # Train model on the partitioned graphs
            for i in range(k):
                part_data = dgl.distributed.load_partition('tmp/test/benchcora.json', i)
                g, nfeat, efeat, partition_book, graph_name, ntypes, etypes = part_data
                train(g, features[g.ndata[dgl.NID]], labels[g.ndata[dgl.NID]], train_mask[g.ndata[dgl.NID]], model)
    return t.elapsed_secs / 3
