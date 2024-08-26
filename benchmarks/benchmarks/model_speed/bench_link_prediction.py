import itertools
import time
import os
import dgl
import dgl.data
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv
import dgl.function as fn
from sklearn.metrics import roc_auc_score
from torch.multiprocessing import Process, Manager

os.environ["DGLBACKEND"] = "pytorch"

from .. import utils

# ----------- 2. create model -------------- #
# build a two-layer GraphSAGE model
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, "mean")
        self.conv2 = SAGEConv(h_feats, h_feats, "mean")
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(fn.u_dot_v("h", "h", "score"))
            return g.edata["score"][:, 0]

class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)
    def apply_edges(self, edges):
        h = torch.cat([edges.src["h"], edges.dst["h"]], 1)
        return {"score": self.W2(F.relu(self.W1(h))).squeeze(1)}
    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(self.apply_edges)
            return g.edata["score"]

def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)

def train_partition(i, k, algorithm, vertex_weight, graph_name, train_g, train_pos_g, train_neg_g, features, model):
    part_data = dgl.distributed.load_partition('tmp/partitioned/' + graph_name + '.json', i)
    g, nfeat, efeat, partition_book, graph_name, ntypes, etypes = part_data

    pred = DotPredictor()
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.01)

    for epoch in range(100):
        # forward
        h = model(g, features[g.ndata[dgl.NID]])

        # Create positive and negative subgraphs for the current partition
        pos_subgraph = dgl.node_subgraph(train_pos_g, g.ndata[dgl.NID])
        neg_subgraph = dgl.node_subgraph(train_neg_g, g.ndata[dgl.NID])

        pos_score = pred(pos_subgraph, h)
        neg_score = pred(neg_subgraph, h)
        loss = compute_loss(pos_score, neg_score)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

@utils.skip_if_gpu()
@utils.benchmark("time", timeout=1200)
@utils.parametrize("graph_name", ["Cora","Citeseer","Pubmed","Reddit"])
@utils.parametrize("vertex_weight",[True,False])
@utils.parametrize("algorithm", [-1,0,1,2,3,4,5])
@utils.parametrize("k", [2, 4, 8])
def track_time(k, algorithm, vertex_weight, graph_name):
    datasets = {
    "Cora": dgl.data.CoraGraphDataset(),
    "Citeseer": dgl.data.CiteseerGraphDataset(),
    "Pubmed": dgl.data.PubmedGraphDataset(),
    "Reddit": dgl.data.RedditDataset(),
    }
    graph = datasets[graph_name][0]
    
    # Split edge set for training and testing
    u, v = graph.edges()
    eids = np.arange(graph.num_edges())
    eids = np.random.permutation(eids)
    test_size = int(len(eids) * 0.1)
    train_size = graph.num_edges() - test_size
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]
    # Find all negative edges and split them for training and testing
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    adj_neg = 1 - adj.todense() - np.eye(graph.num_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)
    neg_eids = np.random.choice(len(neg_u), graph.num_edges())
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]
    train_g = dgl.remove_edges(graph, eids[:test_size])
    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=graph.num_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=graph.num_nodes())
    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=graph.num_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=graph.num_nodes())
    model = GraphSAGE(train_g.ndata["feat"].shape[1], 16)
    features = train_g.ndata['feat']
    score = 0 
    part_time = 0

    with utils.Timer() as t:
        if i == 0:
            if algorithm == -1:
                dgl.distributed.partition_graph(graph, graph_name, k, "tmp/partitioned", part_method="metis", balance_edges=vertex_weight)
            else:
                dgl.distributed.partition_graph(graph, graph_name, k, "tmp/partitioned", part_method="kahip", balance_edges=vertex_weight, mode=algorithm)
            part_time = t.elapsed_secs
                
        processes = []
        for i in range(k):
            p = Process(target=train_partition, args=(i, k, algorithm, vertex_weight, graph_name, train_g, train_pos_g, train_neg_g, features, model))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        #check results #
        pred = DotPredictor()
        with torch.no_grad():
            h = model(train_g, train_g.ndata["feat"])
            pos_score = pred(test_pos_g, h)
            neg_score = pred(test_neg_g, h)
            score += compute_auc(pos_score, neg_score)

    return (t.elapsed_secs-part_time) / 3 , part_time, score / 3
