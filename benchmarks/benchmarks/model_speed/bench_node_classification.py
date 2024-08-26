import time
import os
from multiprocessing import Process

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


def train(g, features, labels, train_mask, test_mask, model, epochs=100, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        logits = model(g, features)
        pred = logits.argmax(1)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return (pred[test_mask] == labels[test_mask]).float().mean()


def train_partition(part_id, graph_name, features, labels, train_mask,test_mask, model):
    # Load the partition
    part_data = dgl.distributed.load_partition('tmp/partitioned/' + graph_name + '.json', part_id)
    g, nfeat, efeat, partition_book, graph_name, ntypes, etypes = part_data
    # Train on the partition
    return train(g, features[g.ndata[dgl.NID]], labels[g.ndata[dgl.NID]], train_mask[g.ndata[dgl.NID]],test_mask[g.ndata[dgl.NID]] model)
    

@utils.skip_if_gpu()
@utils.benchmark("time", timeout=1200)
@utils.parametrize("graph_name", ["Cora","Citeseer","Pubmed"])
@utils.parametrize("vertex_weight",[True,False])
@utils.parametrize("algorithm", [-1,0,1,2,3,4,5])
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
    test_mask = graph.ndata['test_mask']

    # Create model
    model = GCN(graph.ndata['feat'].shape[1], 16, len(torch.unique(labels)))
    # timing
    part_time = 0
    score = 0
    with utils.Timer() as t:
        for i in range(3):
            # Partition the graph
            if i == 0:
                if algorithm == -1:
                    dgl.distributed.partition_graph(graph, graph_name, k, "tmp/partitioned", part_method="metis", balance_edges=vertex_weight)
                else:
                    dgl.distributed.partition_graph(graph, graph_name, k, "tmp/partitioned", part_method="kahip", balance_edges=vertex_weight, mode=algorithm)
                part_time = t.elapsed()
            processes = []
            for part_id in range(k):
                p = Process(target=train_partition, args=(part_id, graph_name, features, labels, train_mask,test_mask, model))
                score += p.start()
                processes.append(p)

            for p in processes:
                p.join()
                
    return (t.elapsed_secs-part_time) / 3, part_time, score/(3*k)
