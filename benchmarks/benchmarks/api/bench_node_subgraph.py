import time

import dgl
import dgl.function as fn

import numpy as np
import torch

from .. import utils


@utils.benchmark("time")
@utils.parametrize("graph_name", ["livejournal", "reddit"])
@utils.parametrize("seed_nodes_num", [200, 5000, 20000])
def track_time(graph_name, seed_nodes_num):
    device = utils.get_bench_device()
    graph = utils.get_graph(graph_name, "coo")
    graph = graph.to(device)
    dgl.distributed.partition_graph(graph,graph_name,4,'tmp/test')

    for j in range(4):
        part_data = dgl.distributed.load_partition('tmp/test/' + graph_name + '.json', j)
        g, nfeat, efeat, partition_book, graph_name, ntypes, etypes = part_data

        seed_nodes = np.random.randint(0, g.num_nodes(), seed_nodes_num)
        seed_nodes = torch.from_numpy(seed_nodes).to(device)

        # dry run
        for i in range(3):
            dgl.node_subgraph(g, seed_nodes)

        # timing
        num_iters = 50
        with utils.Timer() as t:
            for i in range(num_iters):
                dgl.node_subgraph(g, seed_nodes)

        return t.elapsed_secs / num_iters
