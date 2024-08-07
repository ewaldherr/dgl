import time

import dgl

import torch

from .. import utils


def _random_walk(g, seeds, length):
    return dgl.sampling.random_walk(g, seeds, length=length)


def _node2vec(g, seeds, length):
    return dgl.sampling.node2vec_random_walk(g, seeds, 1, 1, length)


@utils.skip_if_gpu()
@utils.benchmark("time")
@utils.parametrize("graph_name", ["cora"])#, "livejournal", "friendster"])
@utils.parametrize("num_seeds", [10, 100, 1000])
@utils.parametrize("length", [2, 5, 10, 20])
@utils.parametrize("vertex_weight",[True,False])
@utils.parametrize("algo",["kahip","metis"])
@utils.parametrize("algorithm", ["_random_walk", "_node2vec"])
def track_time(graph_name, num_seeds, length, algorithm, algo, vertex_weight):
    device = utils.get_bench_device()
    graph = utils.get_graph(graph_name, "coo")
    dgl.distributed.partition_graph(graph,graph_name,4,'tmp/test', part_method = algo, balance_edges = vertex_weight)
    for j in range(4):
        part_data = dgl.distributed.load_partition('tmp/test/' + graph_name + '.json', j)
        g, nfeat, efeat, partition_book, graph_name, ntypes, etypes = part_data
        seeds = torch.randint(0, g.num_nodes(), (num_seeds,))
        print(graph_name, num_seeds, length)
        alg = globals()[algorithm]
        # dry run
        for i in range(5):
            _ = alg(g, seeds, length=length)

        # timing
        with utils.Timer() as t:
            for i in range(50):
                _ = alg(g, seeds, length=length)

        return t.elapsed_secs / 50