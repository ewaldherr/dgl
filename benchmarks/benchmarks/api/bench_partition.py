import time

import dgl

import numpy as np
import torch

from .. import utils


@utils.skip_if_gpu()
@utils.benchmark("time", timeout=1200)
@utils.parametrize("graph_name", ["reddit"])
#@utils.parametrize("algorithm", ["kahip","metis"])
@utils.parametrize("k", [2, 4, 8])
def track_time(graph_name, k):
    device = utils.get_bench_device()
    data = utils.process_data(graph_name)
    graph = data[0]
    # dry run
    # gg = dgl.distributed.partition_graph(graph,graph_name, k,"tmp/test",part_method=algorithm)

    # timing
    with utils.Timer() as t:
        for i in range(3):
            dgl.distributed.partition_graph(graph,graph_name, k,"tmp/test",part_method="metis")

    return t.elapsed_secs / 3
