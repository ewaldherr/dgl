import time

import dgl

import numpy as np
import torch

from .. import utils


@utils.benchmark("time", timeout=60)
@utils.parametrize("graph_name", ["reddit"])
@utils.parametrize("k", [1, 3, 5])
def track_time(graph_name, k):
    device = utils.get_bench_device()
    graph = utils.get_graph(graph_name, "coo").to(device)
    graph = graph.formats("coo")
    dgl.distributed.partition_graph(graph,graph_name,4,'tmp/test')
    for j in range(4):
        part_data = dgl.distributed.load_partition('tmp/test/' + graph_name + '.json', j)
        g, nfeat, efeat, partition_book, graph_name, ntypes, etypes = part_data

        # dry run
        dgl.khop_graph(g, k)

        # timing
        with utils.Timer() as t:
            for i in range(10):
                gg = dgl.khop_graph(g, k)

        return t.elapsed_secs / 10
