import time

import dgl
import dgl.function as fn

import numpy as np
import torch

from .. import utils


@utils.benchmark("time")
@utils.parametrize_cpu("graph_name", ["livejournal"])
@utils.parametrize_gpu("graph_name", ["ogbn-arxiv", "reddit"])
@utils.parametrize("vertex_weight",[True,False])
@utils.parametrize("algorithm",["kahip","metis"])
@utils.parametrize("seed_nodes_num", [200, 5000, 20000])
@utils.parametrize("fanout", [5, 20, 40])
def track_time(graph_name, seed_nodes_num, fanout, algorithm, vertex_weight):
    device = utils.get_bench_device()
    graph = utils.get_graph(graph_name, "coo")

    dgl.distributed.partition_graph(graph,graph_name,4,'tmp/test', part_method = algorithm, balance_edges = vertex_weight)
    edge_dir = "in"
    for j in range(4):
        part_data = dgl.distributed.load_partition('tmp/test/' + graph_name + '.json', j)
        g, nfeat, efeat, partition_book, graph_name, ntypes, etypes = part_data
        g = g.to(device)
        seed_nodes = np.random.randint(0, g.num_nodes(), seed_nodes_num)
        seed_nodes = torch.from_numpy(seed_nodes).to(device)

        # dry run
        for i in range(3):
            dgl.sampling.sample_neighbors_fused(
                g, seed_nodes, fanout, edge_dir=edge_dir
            )

        # timing
        with utils.Timer() as t:
            for i in range(50):
                dgl.sampling.sample_neighbors_fused(
                    g, seed_nodes, fanout, edge_dir=edge_dir
                )

        return t.elapsed_secs / 50