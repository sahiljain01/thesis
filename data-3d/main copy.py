# Goal: Generate data for rigidity analysis through network relaxation (page 115)
# Initial configuration:
#   Randomly generate graph topology
#   Set some interval for x, y, z (-10, 10?)
#   Randomly assign points to each node in the graph

import networkx as nx
import numpy as np
from scipy.linalg import null_space
import random
from tqdm import tqdm

def _open_maybe_compressed(path, mode):
    if path.endswith('.gz'):
        with gzip.open(path, mode) as f:
            yield f
    else:
        with open(path, mode) as f:
            yield f

def random_coordinate(r=10):
    return np.random.uniform(-r, r)

def realize_graph(G, dx=10, dy=10, dz=10):
    # have to assign coordinates in 3-dimensions
    # assign each point coordinates (and calculate edge lengths)
    pos = {}
    for i in range(0, G.number_of_nodes()):
        x, y, z = random_coordinate(), random_coordinate(), random_coordinate()
        pos[i] = [x, y, z]
    return pos
    
def generate_rigidity_matrix(G, config):
    """
    A matrix can be constructed from the entire set of equations of the form of equation (1).
    This matrix, consisting of |E| rows and 3|V | columns (in three dimensions), is called a
    rigidity matrix [22] where each row corresponds to an edge and each column corresponds
    to a coordinate of a vertex.
    """
    pos = config # generate coordinates for graph
    d = 3  # 3-dimensional
    n = G.number_of_nodes()
    e = G.number_of_edges()
    R = np.zeros((e, d * n)) # matrix is dn x e (n: # of vertices, e: number of edges)

    edge_index = 0
    for u, v in G.edges():
        c_u, c_v = pos[u], pos[v]
        c_u_x, c_v_x = c_u[0], c_v[0]       
        c_u_y, c_v_y = c_u[1], c_v[1]
        c_u_z, c_v_z = c_u[2], c_v[2]

        R[edge_index, u*d:u*d+3] = c_u_x - c_v_x, c_u_y - c_v_y, c_u_z - c_v_z
        R[edge_index, v*d:v*d+3] = c_v_x - c_u_x, c_v_y - c_u_y, c_v_z - c_u_z
        edge_index += 1

    ns = null_space(R)
    return (ns.shape[1] == 6) # d(d+1)/2

def add_redudant_edges(G, pos):
    """
    create graph with 24 distinct edges that are non-redudant
    """
    # create set of non-edges and then randomly sample
    mininal_edge_num = 3 * G.number_of_nodes() - (3 * 4 / 2)
    non_edges = set()
    num_nodes = G.number_of_nodes()
    for u in range(0, num_nodes):
        for v in range(u+1, num_nodes):
            if not G.has_edge(u, v):
                non_edges.add((u, v))
    
    # randomly query from set of non-edges until the graph doesn't have 24 edges
    non_edges_sample_order = random.sample(non_edges, len(non_edges))
    
    # add one by one until 24 while maintaining not rigid
    for (u, v) in non_edges_sample_order:
        G.add_edge(u, v)
        if generate_rigidity_matrix(G, pos):
            G.remove_edge(u, v)
        
        if G.number_of_edges() == mininal_edge_num:
            break 

    return G

def create_minimally_rigid_graph(G, pos):
    mininal_edge_num = 3 * G.number_of_nodes() - (3 * 4 / 2)
    print(f"this graph has: {G.number_of_edges()}, a minimally rigid graph would have: {mininal_edge_num} edges")
    # intelligent edge removal 
    while (G.number_of_edges() != mininal_edge_num):
        for u, v in G.edges():
            G.remove_edge(u, v)
            if not generate_rigidity_matrix(G, pos):
                G.add_edge(u, v)
    return G

def generate_random_graph(n, p):
    new_graph = nx.erdos_renyi_graph(n, p, seed=None, directed=False)
    return new_graph

if __name__ == "__main__":
    rigid_graphs = []
    flexible_graphs = []

    total_rigid = 0
    total_not_rigid = 0
    for i in tqdm(np.arange(0.1, 1.0, 0.1), desc="outer"):
        for j in tqdm(range(0, 10), desc="inner"):
            num_nodes = random.randint(50, 70)
            new_graph = generate_random_graph(num_nodes, i)
            # print("************************************")
            # print(f"creating a graph with size: {num_nodes} and density: {i}")
            pos1 = nx.spring_layout(new_graph, dim=3)
            pos2 = realize_graph(new_graph)
            
            if generate_rigidity_matrix(new_graph, pos2) and generate_rigidity_matrix(new_graph, pos1):
                create_minimally_rigid_graph(new_graph, pos2)
                total_rigid += 1
                # print("minimally rigid: ", new_graph.number_of_edges())
                rigid_graphs.append(new_graph)
            else:
                add_redudant_edges(new_graph, pos1)
                total_not_rigid += 1
                # print("not minimally rigid: ", new_graph.number_of_edges())
                flexible_graphs.append(new_graph)
            # print("************************************")

    print("Summary Statistics:")
    print("Number of rigid graphs: ", total_rigid, ". Number of flexible graphs: , ", total_not_rigid)

    filename = "data/test.pkl"
    if filename is not None:
        with _open_maybe_compressed(filename, 'wb') as f:
            pickle.dump((rigid_graphs, flexible_graphs), f, pickle.HIGHEST_PROTOCOL)