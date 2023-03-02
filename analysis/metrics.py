
from numpy import mean
from igraph import Graph, rescale
# from file_generator import add_sheet_to_xlsx
# from plots import plot_snapshot_metrics
import pandas as pd


def betweenness(g=Graph, average=False):
    if (average):
        return mean(rescale(g.betweenness(directed=True)))
    return rescale(g.betweenness(directed=True))


def degree(g=Graph, average=False):
    if (average):
        return mean(g.degree(mode='all'))
    return g.degree(mode='all')


def closeness(g=Graph, average=False):
    if (average):
        return mean(g.closeness(mode='all'))
    return g.closeness(mode='all')


def mincut(g=Graph, average=False):
    return g.mincut().value


def edge_betweenness(g=Graph, average=False):
    if (average):
        return mean(g.edge_betweenness())
    return g.edge_betweenness()


def clustering_coefficient(g=Graph, average=False):
    if (average):
        return mean(g.transitivity_undirected())
    return g.transitivity_undirected()


def pagerank(g=Graph, average=False):
    if (average):
        return mean(g.pagerank(directed=True))
    return g.pagerank(directed=True)


def compute_average_metrics(g=Graph):

    return [degree(g=g, average=True), betweenness(
        g=g, average=True), closeness(g=g, average=True), pagerank(g=g, average=True), clustering_coefficient(
        g=g, average=True), mean(g.eccentricity()),
        mincut(g), edge_betweenness(g, average=True)]


def compute_graph_metrics(g=Graph):

    data = {}

    data['Degree'] = degree(g=g)
    data['Betweenness'] = betweenness(
        g=g)
    data['Closeness'] = closeness(g=g)
    data['Page Rank'] = pagerank(g=g)
    data['Clustering Coefficient'] = clustering_coefficient(
        g=g)
    data["Revenu"] = g.vs['revenu']
    data["Age"] = g.vs['age']
    data["Accorderie"] = g.vs['accorderie']
    data["Ville"] = g.vs['ville']
    data["Region"] = g.vs['region']
    data["Arrondissement"] = g.vs['arrondissement']

    result = pd.DataFrame(
        data=data)

    return result


def compute_global_properties_on_graph(g=Graph):
    columns = ['Value']
    indices = ['Diameter', 'Radius', 'Density',
               'Average path length', 'Girth', 'Reciprocity', 'Eccentricity', 'Clustering coefficient', 'Edge betweenness']
    data = [g.diameter(directed=True), g.radius(), g.density(),
            g.average_path_length(directed=True), g.girth(
    ), g.reciprocity(), mean(g.eccentricity()),
        clustering_coefficient(g, average=True), edge_betweenness(g, average=True)]

    result = pd.DataFrame(
        data=data, index=indices, columns=columns)

    return result


def compute_degree_distribution(g=Graph, degree_mode='out'):
    xa, ya = zip(*[(int(left), count) for left, _, count in
                   g.degree_distribution(mode=degree_mode).bins()])

    return xa, ya
