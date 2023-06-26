
from numpy import mean
from igraph import Graph
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
        return mean(g.betweenness(directed=True, weights=weights))
    return g.betweenness(directed=True, weights=weights)


def degree(g=Graph, average=False):
    if (average):
        return mean(g.degree(mode='all', loops=True))
    return g.degree(mode='all', loops=True)


def closeness(g=Graph, average=False, weights=[]):
    if (average):
        return mean(g.closeness(mode='all', weights=weights))
    return g.closeness(mode='all', weights=weights)


def harmonic(g=Graph, average=False, weights=[]):
    if (average):
        return mean(g.harmonic_centrality(mode='all', weights=weights))
    return g.harmonic_centrality(mode='all', weights=weights)


def katz(g=Graph, average=False, weights=[]):
    if (average):
        return mean(g.harmonic_centrality(mode='all', weights=weights))
    return g.harmonic_centrality(mode='all', weights=weights)


def eigencentrality(g=Graph, average=False, weights=[]):
    if (average):
        return mean(g.eigenvector_centrality(directed=True, weights=weights))
    return g.eigenvector_centrality(directed=True, weights=weights)


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


def centralization(n, metrics=[], metric_name=''):
    denominator_factor = None

    if not metric_name:
        return np.nan

    if metric_name == 'degree':
        denominator_factor = n**2 - 3*n + 2
    elif metric_name == 'betweenness':
        denominator_factor = n**3 - 4*n**2 + 5*n - 2
    elif metric_name == 'harmonic':
        denominator_factor = (2*n**2 - 6*n + 4)/(2*n - 3)
    elif metric_name == 'closeness':
        denominator_factor = (n**2 - 3*n + 2)/(2*n - 3)

    if not denominator_factor:
        return np.nan

    return np.sum((np.max(metrics) - np.array(metrics))) / denominator_factor


def compute_average_metrics(g=Graph):
    weights = compute_edge_weight_based_on_edge_number(g)

    averages = [
        degree(g=g, average=True), 
        g.maxdegree(mode='in'), 
        g.maxdegree(mode='out'), 
        betweenness(g=g, average=True, weights=weights), 
        closeness(g, average=True, weights=weights), 
        harmonic(g, average=True, weights=weights),  
        pagerank(g=g, average=True, weights=weights), 
        clustering_coefficient(g=g, average=True), 
        global_clustering_coefficient(g), 
        edge_betweenness(g, average=True, weights=weights)
    ]

    structural = global_graph_properties(g)

    return np.concatenate([averages, structural])


def compute_graph_metrics(g=Graph):
    weights = compute_edge_weight_based_on_edge_number(g)
    data = {}

    data['Degree'] = degree(g=g)
    data['Betweenness'] = betweenness(g=g, weights=weights)
    data['Closeness'] = closeness(g=g, weights=weights)
    data['Harmonic'] = harmonic(g=g, weights=weights)
    data['Eccentricity'] = g.eccentricity()
    # data['Eigenvector centrality'] = pagerank(g=g, weights=weights)
    data['Page Rank'] = pagerank(g=g, weights=weights)
    data['Local clustering coefficient'] = clustering_coefficient(g=g)

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
    weights = compute_edge_weight_based_on_edge_number(g)

    n = len(g.vs)

    weak_size, strong_size = giant_component(g)

    data = [
        g.vcount(),
        g.ecount(),
        g.maxdegree(mode='in', loops=True),
        g.maxdegree(mode='out', loops=True),
        mean_degree(g),
        get_avg_in_out_degree(g),
        get_avg_weighted_in_out_degree(g, field_name='duree'),
        get_avg_in_out_disbalance(g),
        get_unique_edges_vs_total(g),
        g.diameter(directed=True),
        g.radius(),
        g.density(),
        g.average_path_length(directed=True),
        g.reciprocity(ignore_loops=False),
        mean(g.eccentricity()),
        weak_size,
        strong_size,
        power_law_alpha(g),
        global_clustering_coefficient(g),
        clustering_coefficient(g, average=True),
        centralization(n, degree(g), 'degree'),
        centralization(n, betweenness(g, weights=weights), 'betweenness'),
        centralization(n, closeness(g, weights=weights), 'closeness'),
        centralization(n, harmonic(g, weights=weights), 'harmonic'),
        pagerank(g, average=True, weights=weights),
        degree_assortativity(g),
        homophily_nominal(g, 'age'),
        homophily_nominal(g, 'revenu'),
        homophily_nominal(g, 'ville'),
        homophily_nominal(g, 'region'),
        homophily_nominal(g, 'arrondissement'),
        homophily_nominal(g, 'adresse'),
    ]

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


'''
    DEPRECATED: replaced with average_path_lenght from igraph
'''
# def small_world_mean_distance(g):
#     shortest_paths_pairs = pd.DataFrame(g.shortest_paths())

#     shortest_paths_pairs.replace([np.inf, -np.inf], 0, inplace=True)

#     total_distance = sum(sum(row) for row in shortest_paths_pairs.values)

#     l = total_distance / g.vcount()**2
#     log_mean_dist = np.log10(g.vcount())
#     print(g.vcount())
#     return l, log_mean_dist


def giant_component(g):
    s_components = g.connected_components(mode='strong')
    w_components = g.connected_components(mode='weak')
    # print(s_components)
    s = len(max(s_components, key=len)) / \
        g.vcount()  # giant strongly component
    w = len(max(w_components, key=len)) / \
        g.vcount()  # giant weakly component

    return w, s


def power_law_alpha(g):
    degrees = g.degree()

    fit = pl.Fit(degrees, verbose=False)

    alpha = fit.alpha

    return alpha


def global_clustering_coefficient(g):
    # l_cc = g.transitivity_undirected(mode="zero")

    g_cc = g.transitivity_avglocal_undirected(mode="zero")

    return g_cc


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



def duree_to_int(duree_str):
    ret = 0
    if duree_str and ':' in duree_str:
        pz = duree_str.split(":")
        ret += float(pz[0])
        ret += float(pz[1]) / 60 * 100
    return ret


# average indeg / (indeg + outdeg).  Less than 0.5 => outdeg bias, higher => indeg bias
def get_avg_in_out_degree(g):
    
    if len(g.vs) == 0:
        return -1
    
    ratio_sum = 0
    nb_isolated = 0
    
    for v in g.vs:
        indeg = g.degree(v, mode='in')
        outdeg = g.degree(v, mode='out')
        
        if outdeg == 0 and indeg == 0:
            nb_isolated += 1
        else:
            ratio_sum += (indeg / (outdeg + indeg))
    
    return ratio_sum / (len(g.vs) - nb_isolated) 







# average duree_in / (duree_in + duree_out).  Less than 0.5 => outdeg bias, higher => indeg bias
def get_avg_weighted_in_out_degree(g, field_name='duree'):
    
    if len(g.vs) == 0:
        return -1
    
    ratio_sum = 0
    nb_isolated = 0
    
    for v in g.vs:
        weight_in = 0
        for e in g.es[g.incident(v, mode='in')]:
            weight_in += duree_to_int(e['duree'])
        
        weight_out = 0
        for e in g.es[g.incident(v, mode='out')]:
            weight_out += duree_to_int(e['duree'])
        
        
        
        if weight_in == 0 and weight_out == 0:
            nb_isolated += 1
        else:
            ratio_sum += (weight_in / (weight_in + weight_out))
        
    
    return ratio_sum / (len(g.vs) - nb_isolated) 










# average max of indeg / (indeg + outdeg) or 1 - that qty.  Minimum is 0.5, closer to 1 => quite disbalanced
def get_avg_in_out_disbalance(g):
    
    if len(g.vs) == 0:
        return -1
    
    disbalance_sum = 0
    nb_isolated = 0
    for v in g.vs:
        indeg = g.degree(v, mode='in')
        outdeg = g.degree(v, mode='out')
        
        if outdeg == 0 and indeg == 0:
            nb_isolated += 1
        else:
            disbalance_sum += max(indeg / (indeg + outdeg), 1 - indeg/(indeg + outdeg))
    
    return disbalance_sum / (len(g.vs) - nb_isolated) 






# ratio of unique edges / edges.  Under 1 => edges are repeated
def get_unique_edges_vs_total(g):
    
    if len(g.es) == 0:
        return -1
    
    nb_edges = len(g.es)
    
    unique_edges = set()
    all_edges = list()
    
    for e in g.es:
        et = e.tuple
        if et not in unique_edges:
            unique_edges.add(et)
            
        all_edges.append(et)

    #all_edges.sort()
    #print(all_edges)

            
    return len(unique_edges) / nb_edges

