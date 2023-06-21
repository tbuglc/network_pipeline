
from numpy import mean
from igraph import Graph, rescale
import pandas as pd
import numpy as np
import powerlaw as pl
from utils import global_graph_indices

def compute_edge_weight_based_on_edge_number(g):
    weights = []
    for e in g.es:
        src, tgt = e.source, e.target
        count = g.count_multiple([(src, tgt)])
        weights = weights + count
        
    return weights

def betweenness(g=Graph, average=False, weights=[]):
    if (average):
        return mean(rescale(g.betweenness(directed=True, weights=weights)))
    return rescale(g.betweenness(directed=True, weights=weights))


def degree(g=Graph, average=False):
    if (average):
        return mean(g.degree(mode='all', loops=True))
    return g.degree(mode='all', loops=True)


def closeness(g=Graph, average=False, weights=[]):
    if (average):
        return mean(g.closeness(mode='all', normalized=True, weights=weights))
    return g.closeness(mode='all', normalized=True, weights=weights)


def mincut(g=Graph, average=False):
    return g.mincut().value


def edge_betweenness(g=Graph, average=False, weights=[]):
    if (average):
        return mean(g.edge_betweenness(directed=True, weights=weights))
    return g.edge_betweenness(directed=True, weights=weights)


def clustering_coefficient(g=Graph, average=False):
    if (average):
        return mean(g.transitivity_undirected(mode='zero'))
    return g.transitivity_undirected(mode='zero')


def pagerank(g=Graph, average=False,  weights=[]):
    if (average):
        return mean(g.pagerank(directed=True, weights=weights))
    return g.pagerank(directed=True, weights=weights)


def compute_average_metrics(g=Graph):
    weights  = compute_edge_weight_based_on_edge_number(g)
    
    return [degree(g=g, average=True), betweenness(
        g=g, average=True, weights=weights), closeness(g=g, average=True, weights=weights), pagerank(g=g, average=True, weights=weights), clustering_coefficient(
        g=g, average=True), mean(g.eccentricity()),
        mincut(g), edge_betweenness(g, average=True, weights=weights)]


def compute_graph_metrics(g=Graph):
    weights  = compute_edge_weight_based_on_edge_number(g)
    data = {}
    data['Degree'] = degree(g=g)
    data['Betweenness'] = betweenness(g=g, weights=weights)
    data['Closeness'] = closeness(g=g, weights=weights)
    data['Page Rank'] = pagerank(g=g, weights=weights)
    data['Clustering Coefficient'] = clustering_coefficient(g=g)


    data["Revenu"] = g.vs['revenu']
    data["Age"] = g.vs['age']
    data["Genre"] = g.vs['genre']
    data["Accorderie"] = g.vs['accorderie']
    data["Ville"] = g.vs['ville']
    data["Region"] = g.vs['region']
    data["Arrondissement"] = g.vs['arrondissement']

    result = pd.DataFrame(
        data=data)

    return result

def global_graph_properties(g=Graph):
    x0 = g.vcount()
    x10 = g.ecount()
    x1 = g.diameter(directed=True)
    x2 = g.radius()
    x3 = g.density()
    x4 = g.average_path_length(directed=True)
    x6 = g.reciprocity()
    x7 = mean(g.eccentricity())
    
    weights  = compute_edge_weight_based_on_edge_number(g)
    
    x8 = clustering_coefficient(g, average=True)
    x9 = edge_betweenness(g, average=True, weights=weights)


    data = [x0, x10,x1, x2, x3, x4,  x6, x7, x8, x9]

    return data

def compute_global_properties_on_graph(g=Graph):
    columns = ['Value']

    data = global_graph_properties(g)

    result = pd.DataFrame(
        data=data, index=global_graph_indices, columns=columns)

    return result


def compute_degree_distribution(g=Graph, degree_mode='out'):
    xa, ya = zip(*[(int(left), count) for left, _, count in
                   g.degree_distribution(mode=degree_mode).bins()])

    return xa, ya


def mean_degree(g):
    return g.ecount() / g.vcount()

def small_world_mean_distance(g):
    shortest_paths_pairs = pd.DataFrame(g.shortest_paths())
    
    shortest_paths_pairs.replace([np.inf, -np.inf], 0, inplace=True)

    total_distance = sum(sum(row) for row in shortest_paths_pairs.values)

    l = total_distance / g.vcount()**2
    log_mean_dist = np.log10(g.vcount())
    print(g.vcount())
    return l, log_mean_dist  


def giant_component(g):
    print(g.vcount(), g.ecount())
    s_components = g.connected_components(mode='strong')
    w_components = g.connected_components(mode='weak')
    # print(s_components)
    S_s = len(max(s_components, key=len)) / g.vcount() # giant strongly component
    S_w = len(max(w_components, key=len)) / g.vcount() # giant weakly component

    return S_s, S_w

def power_law_alpha(g):
    degrees = g.degree()

    fit = pl.Fit(degrees)

    alpha = fit.alpha

    return alpha


def clustering_coefficient(g):
    l_cc = g.transitivity_undirected(mode="zero")

    g_cc = g.transitivity_avglocal_undirected(mode="zero")

    return l_cc, g_cc

def degree_correlation(g):

    degrees = g.degree()

    # Initialize an empty list to store the degrees of connected vertices
    connected_degrees = []

    # Iterate over the edges of the graph
    for edge in g.es():
        source_degree = degrees[edge.source]
        target_degree = degrees[edge.target]
        connected_degrees.append((source_degree, target_degree))

    # Calculate the degree correlation coefficient
    d_c = np.corrcoef(connected_degrees, rowvar=False)[0, 1]

    return d_c

def degree_assortativity(g):
    return g.assortativity_degree()


def homophily_nominal(g, attribute):

    if type(g.vs[attribute][0]) == int:
        return g.assortativity_nominal(attribute, directed=True)


    for v in g.vs:
        # print(v[attribute])
        if type(v[attribute]) == float or type(v[attribute]) == int:
            v[attribute] = 'other'
        
    uni_val = np.unique(g.vs[attribute])

    mapping = {}
    for idx, uv in enumerate(uni_val):
        mapping[uv] = idx

    types = [mapping[att] for att in g.vs[attribute]]

    numeric_label = f'{attribute}_numeric_types'
    
    g.vs[numeric_label] = types


    assortativity = g.assortativity_nominal(numeric_label, directed=True)

    return assortativity
