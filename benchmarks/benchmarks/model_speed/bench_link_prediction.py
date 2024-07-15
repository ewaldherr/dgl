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


@utils.skip_if_gpu()
@utils.benchmark("time", timeout=1200)
@utils.parametrize("graph_name", ["Cora","Citeseer","Pubmed"])
@utils.parametrize("vertex_weight",[True,False])
@utils.parametrize("algorithm", ["kahip","metis"])
@utils.parametrize("k", [2, 4, 8])
#@utils.parametrize("kahip_mode", [1,3])
def track_time(k, algorithm, vertex_weight, graph_name):
    datasets = {
    "Cora": dgl.data.CoraGraphDataset(),
    "Citeseer": dgl.data.CiteseerGraphDataset(),
    "Pubmed": dgl.data.PubmedGraphDataset(),
    }
    graph = datasets[graph_name][0]
    features = graph.ndata['feat']

    # Split edge set for training and testing
    u, v = g.edges()
    eids = np.arange(g.num_edges())
    eids = np.random.permutation(eids)
    test_size = int(len(eids) * 0.1)
    train_size = g.num_edges() - test_size
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

    # Find all negative edges and split them for training and testing
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    adj_neg = 1 - adj.todense() - np.eye(g.num_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)
    neg_eids = np.random.choice(len(neg_u), g.num_edges())
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]

    train_g = dgl.remove_edges(g, eids[:test_size])
    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.num_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.num_nodes())
    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.num_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.num_nodes())

    model = GraphSAGE(train_g.ndata["feat"].shape[1], 16)
    pred = DotPredictor()

    with utils.Timer() as t:
        for i in range(3):
            # ----------- Partition the graph -------------- #
            dgl.distributed.partition_graph(train_g,graph_name, k,"tmp/partitioned",part_method = algorithm, balance_edges = vertex_weight, mode = 0)

            # ----------- 3. set up loss and optimizer -------------- #
            optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.01)

            # ----------- 4. training -------------------------------- #
            all_logits = []
            for epoch in range(100):
                for i in range(k):
                    part_data = dgl.distributed.load_partition('tmp/partitioned/' + graph_name + '.json', i)
                    g, nfeat, efeat, partition_book, graph_name, ntypes, etypes = part_data
                    # forward
                    h = model(g, features[g.ndata[dgl.NID]])
                    pos_score = pred(train_pos_g, h)
                    neg_score = pred(train_neg_g, h)
                    loss = compute_loss(pos_score, neg_score)

                    # backward
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # ----------- 5. check results ------------------------ #
            with torch.no_grad():
                h = model(train_g, train_g.ndata["feat"])
                pos_score = pred(test_pos_g, h)
                neg_score = pred(test_neg_g, h)
                print("AUC", compute_auc(pos_score, neg_score))