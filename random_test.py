import networkx as nx
import numpy as np
import powerlaw as pl
import igraph as ig 
from structure import mean_degree, degree_assortativity, giant_component,clustering_coefficient


p = np.arange(0.015, 0.02 + 0.001, 0.001)
pc = np.arange(0, 1 + 0.1, 0.1)
m =[9,10,11,12]
number_of_nodes = 631


def networkx_to_igraph(g, directed=False):
    ig_graph = ig.Graph(directed=directed)
    # print(g.nodes(), g.edges())
    ig_graph.add_vertices(g.nodes())
    ig_graph.add_edges(g.edges())
    
    return ig_graph


def power_law_alpha(degrees):
    fit = pl.Fit(degrees)
    alpha = fit.alpha

    return alpha

def degree_distribution(g):
    degrees = [ d for n, d in g.degree()]
    degrees.sort()

    return degrees





def random_grph():
    graphs = []
    for v in p:
        g = nx.gnp_random_graph(number_of_nodes, v, directed=True)
        graphs.append(g)
        degrees = degree_distribution(g)
        alpha = power_law_alpha(degrees)

        print(f'Prob: {v}, Minimum degree: {degrees[0]}, Number of edges {g.number_of_edges()}, Alpha: {alpha}')

    return graphs
    
def power_law_cluster():
    graphs = []
    for v in pc:
        for mo in m:
            g = nx.powerlaw_cluster_graph(592, mo, v, 0)
            graphs.append(g)
            degrees = degree_distribution(g)
            alpha = power_law_alpha(degrees)

            print(f'Prob: {v}, Minimum degree: {degrees[0]}, Number of edges {g.number_of_edges()}, Alpha: {alpha}')
    return graphs

def power_law():
    graphs = []
    for mo in m:
        g = nx.barabasi_albert_graph(592, mo)
        graphs.append(g)
        degrees = degree_distribution(g)
        alpha = power_law_alpha(degrees)

        print(f'Minimum degree: {degrees[0]}, Number of edges {g.number_of_edges()}, Alpha: {alpha}')

    return graphs

def small_world():
    graphs = []
    for mo in m:
        g = nx.barabasi_albert_graph(592, mo)
        graphs.append(g)
        degrees = degree_distribution(g)
        alpha = power_law_alpha(degrees)

        print(f'Minimum degree: {degrees[0]}, Number of edges {g.number_of_edges()}, Alpha: {alpha}')

    return graphs





def compute_structural_properties(results, gs):
        
    for g in gs:
        g = networkx_to_igraph(g)
        # print(g)
        _, weakly = giant_component(g)
        l_cc, g_cc = clustering_coefficient(g)

        results.append([g.vcount(), g.ecount(), mean_degree(g), weakly, power_law_alpha(g), l_cc, g_cc, degree_assortativity(g)])

    return results

def generator():
    results = []
    
    rnd_graphs = random_grph()
    results = compute_structural_properties(results=results, gs=rnd_graphs)

    rnd_graphs = power_law()
    results = compute_structural_properties(results=results, gs=rnd_graphs)

    rnd_graphs = small_world()
    results = compute_structural_properties(results=results, gs=rnd_graphs)

    rnd_graphs = power_law_cluster()
    results = compute_structural_properties(results=results, gs=rnd_graphs)
   
    print(results)
    # data frame
    # excel 

generator()