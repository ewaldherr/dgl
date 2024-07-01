import time

import dgl

import numpy as np
import torch

from .. import utils


@utils.skip_if_gpu()
@utils.benchmark("time", timeout=1200)
@utils.parametrize("graph_name", ["cora","livejournal","ogbn-arxiv","friendster","pubmed","ogbn-products","reddit"])
@utils.parametrize("k", [4])
def track_time(graph_name, k):
    device = utils.get_bench_device()
    graph = utils.get_graph(graph_name, "coo")
    graph = graph.to(device)

    # timing
    with utils.Timer() as t:
        for i in range(3):
            gg = dgl.transforms.kahip_partition_assignment(graph,k)

    return t.elapsed_secs / 3
