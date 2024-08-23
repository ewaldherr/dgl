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


os.environ["DGLBACKEND"] = "pytorch"

from .. import utils

@utils.skip_if_gpu()
@utils.benchmark("time", timeout=1200)
@utils.parametrize("graph_name", ["Hollywood2011"])
def track_time(k, algorithm, vertex_weight, graph_name):
    datasets = {
    "Hollywood2011": dgl.data.Hollywood2011Dataset(),
    }
    graph = datasets[graph_name][0]
    
    

    with utils.Timer() as t:
        for i in range(3):
            print(f"Graph has {graph.num_nodes()} nodes and {graph.num_edges()} edges")
    return t.elapsed_secs / 3 , score / 3
