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

def train(g, features, labels, train_mask, model, epochs=30, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        logits = model(g, features)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def train_partition(part_id, graph_name, features, labels, train_mask, model, tmp_dir):
    # Load the partition
    part_data = dgl.distributed.load_partition(tmp_dir + '/partitioned/' + graph_name + '.json', part_id)
    g, nfeat, efeat, partition_book, graph_name, ntypes, etypes = part_data
    # Train on the partition
    train(g, features[g.ndata[dgl.NID]], labels[g.ndata[dgl.NID]], train_mask[g.ndata[dgl.NID]], model)
    print(f"Finish training partition {part_id}")

@utils.skip_if_gpu()
@utils.benchmark("time", timeout=10620)
@utils.parametrize("graph_name", ["Yelp"])
@utils.parametrize("k", [32])
@utils.parametrize("vertex_weight",[True])
@utils.parametrize("algorithm", [5])
def track_time(k, algorithm, vertex_weight, graph_name):
    #Save graph data to $TMPDIR of used node
    tmp_dir = os.getenv('TMPDIR', '~/.dgl')
    datasets = {
        "Yelp": dgl.data.YelpDataset(raw_dir = tmp_dir),
    }
    graph = datasets[graph_name][0]

    # Get features and labels
    features = graph.ndata['feat']
    labels = graph.ndata['label']
    train_mask = graph.ndata['train_mask']
    # Create model
    model = GCN(graph.ndata['feat'].shape[1], 16, len(torch.unique(labels)))
    # Partition the graph
    if algorithm == -1:
        dgl.distributed.partition_graph(graph, graph_name, k, tmp_dir +"/partitioned", part_method="metis", balance_edges=vertex_weight)
    else:
        dgl.distributed.partition_graph(graph, graph_name, k, tmp_dir +"/partitioned", part_method="kahip", balance_edges=vertex_weight, mode=algorithm)
    
    # timing
    with utils.Timer() as t:
        for i in range(1):
            processes = []
            for part_id in range(k):
                p = Process(target=train_partition, args=(part_id, graph_name, features, labels, train_mask, model, tmp_dir))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()
                
    return t.elapsed_secs / 1
