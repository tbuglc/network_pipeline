
from numpy import mean
from igraph import Graph, rescale
import pandas as pd
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
    x0=g.vcount()
    x10 = g.ecount()
    x1 = g.diameter(directed=True)
    x2 = g.radius()
    x3 = g.density()
    x4 = g.average_path_length(directed=True)
    # x5 = g.girth()
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
