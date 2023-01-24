import pickle
import contextlib
import numpy as np
import networkx as nx

from random import random
from tqdm import trange

import gzip
from copy import deepcopy

import pyximport; pyximport.install()
from graph import Graph

from laman_edit import generate_random_HI, generate_random_HII, compute_HI, compute_HII, is_laman_custom, apply_random_corruptions

# degree of decomposibility 
_dod_p_dist = {
    'low': lambda rng: rng.uniform(0.0, 0.1),
    'med': lambda rng: rng.uniform(0.4, 0.6),
    'high': lambda rng: rng.uniform(0.9, 1.0)
}

@contextlib.contextmanager
def _open(path, mode):
    assert path.endswith('.gz'), "file path must end in .gz!"
    with gzip.open(path, mode) as f:
        yield f

def generate_graph(num_nodes, p_type1, rng=np.random):
    """ Generate random graph based on n, the number of vertices, 
    and p, the probability of choosing Henneberg type I / type II operation.
    """
    # Initialize adjacency matrix to a triangle
    G = nx.Graph()
    for i in range(3):
        for j in range(i+1, 3):
            G.add_edge(i, j)

    # Connect additional edges via HI or HII
    for i in range(3, num_nodes):
        if rng.uniform() < p_type1:
            act = generate_random_HI(G, rng=rng)
            G = compute_HI(G, act)
        else:
            act = generate_random_HII(G, rng=rng)
            if act is None:
                # couldn't find another node pair
                return G
            else:
                G = compute_HII(G, act)
    return G

def generate_dataset(num_graphs, size_dist=None, p_dist=None, rng=np.random, filename=None):
    if not size_dist:
        def size_dist(rng):
            return int(np.round(rng.normal(loc=30, scale=30)))
    if not p_dist:
        def p_dist(rng):
            return rng.uniform(low=0.4, high=0.6)

    laman_data = []
    not_laman_data = []
    for i in trange(num_graphs):
        this_size = size_dist(rng)
        while (this_size < 3):
            this_size = size_dist(rng)

        this_p = p_dist(rng)
        laman_g = generate_graph(this_size, this_p, rng=rng)  
        if laman_g is None:      
            continue 

        laman_edges = laman_g.edges
        not_laman_g = deepcopy(laman_g)

        for laman_edge in laman_edges:
            inc = rng.uniform(low=0.0, high=1.0)
            if inc > 0.9:
                if not_laman_g is not None:
                    not_laman_g.remove_edge(laman_edge[0], laman_edge[1])                
        
        if not_laman_g is None:
            continue 

        not_laman_g.remove_nodes_from(list(nx.isolates(not_laman_g)))

        p = rng.uniform(0.0, 1.0)
        if p >= 0.5:
            laman_data.append(laman_g)
        else:
            not_laman_data.append(not_laman_g)



    if filename is not None:
        with _open(filename, 'wb') as f:
            pickle.dump((laman_data, not_laman_data), f, pickle.HIGHEST_PROTOCOL)

    return laman_data


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',
        default='data.gz',
        help='The output file. If it ends in .gz, will be compressed using gzip')

    parser.add_argument('--num-samples', type=int, default=20000)
    parser.add_argument('--dod', choices=['low', 'med', 'high'], default='low',
        help='The distribution of the degree of decomposability.')

    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        rng = np.random.RandomState(args.seed)
    else:
        rng = np.random

    generate_dataset(args.num_samples, filename=args.output, p_dist=_dod_p_dist[args.dod], rng=rng)


if __name__ == '__main__':
    main()