import time

import dgl

import numpy as np
import torch

from .. import utils


@utils.skip_if_gpu()
@utils.benchmark("time", timeout=1200)
@utils.parametrize("graph_name", ["cora","pubmed"])
@utils.parametrize("vertex_weight",[True,False])
@utils.parametrize("algorithm", ["kahip","metis"])
@utils.parametrize("k", [2, 4, 8])
def track_time(graph_name, k, algorithm, vertex_weight):
    device = utils.get_bench_device()
    data = utils.process_data(graph_name)
    graph = data[0]

    # timing
    with utils.Timer() as t:
        for i in range(3):
            dgl.distributed.partition_graph(graph,graph_name, k,"tmp/test",part_method = algorithm, balance_edges = vertex_weight)

    return t.elapsed_secs / 3
