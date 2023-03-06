
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
    print('Computed degree centrality')
    data['Betweenness'] = betweenness(
        g=g)
    print('Computed betweenness centrality')
    data['Closeness'] = closeness(g=g)
    print('Computed closeness centrality')
    data['Page Rank'] = pagerank(g=g)
    print('Computed page rank centrality')
    data['Clustering Coefficient'] = clustering_coefficient(
        g=g)
    print('Computed clustering coefficient')

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


    print('global metrics')
    x1 = g.diameter(directed=True)
    print('diameter')
    x2 = g.radius()
    print('radius')
    x3 = g.density()
    print('density')
    x4 = g.average_path_length(directed=True)
    print('average path length')
    x5 = g.girth()
    print('girth')
    x6 = g.reciprocity()
    print('reciprocity')
    x7 = mean(g.eccentricity())
    print('eccentricity')
    x8 = clustering_coefficient(g, average=True)
    print('cc')
    x9 = edge_betweenness(g, average=True)
    print('edge betweeness')


    data = [x1, x2, x3, x4, x5, x6, x7, x8, x9]

    result = pd.DataFrame(
        data=data, index=indices, columns=columns)

    return result


def compute_degree_distribution(g=Graph, degree_mode='out'):
    xa, ya = zip(*[(int(left), count) for left, _, count in
                   g.degree_distribution(mode=degree_mode).bins()])

    return xa, ya
