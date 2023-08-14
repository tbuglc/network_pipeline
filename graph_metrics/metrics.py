
from numpy import mean
from igraph import Graph, plot
import pandas as pd
import numpy as np
import powerlaw as pl
from utils import global_graph_indices
from dateutil import parser
import sys
from matplotlib.backends.backend_pdf import PdfPages

from datetime import datetime, timedelta
import os
# graph_filter_path = 'C:\\Users\\bugl2301\\projects\\school\\network_pipeline'
# graph_common_path = 'C:\\Users\\bugl2301\\projects\\school\\network_pipeline\\graph'

# if graph_filter_path not in sys.path:
#     sys.path.append(graph_filter_path)
#     # print(sys.path)

from utils import perform_filter_on_graph

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
    if len(weights) > 0:
        return g.betweenness(directed=True, weights=weights)
    return g.betweenness(directed=True)


def degree(g=Graph, average=False, loops=False, mode='all'):
    if (average):
        return mean(g.degree(mode=mode, loops=loops))
    return g.degree(mode=mode, loops=loops)


def closeness(g=Graph, average=False, weights=[], mode='all'):
    if (average):
        return mean(g.closeness(mode=mode, weights=weights))
    if len(weights) > 0:
        return g.closeness(mode=mode, weights=weights)
    return g.closeness(mode=mode)


def harmonic(g=Graph, average=False, weights=[], mode='all'):
    if (average):
        return mean(g.harmonic_centrality(mode=mode, weights=weights))
    if len(weights) > 0:
        return g.harmonic_centrality(mode=mode, weights=weights)

    return g.harmonic_centrality(mode=mode)


def katz(g=Graph, average=False, weights=[], mode='all'):
    if (average):
        return mean(g.harmonic_centrality(mode=mode, weights=weights))
    if len(weights) > 0:
        return g.harmonic_centrality(mode=mode, weights=weights)

    return g.harmonic_centrality(mode=mode)


def eigencentrality(g=Graph, average=False, weights=[]):
    if (average):
        return mean(g.eigenvector_centrality(directed=True, weights=weights))
    if len(weights) > 0:
        return g.eigenvector_centrality(directed=True, weights=weights)

    return g.eigenvector_centrality(directed=True)


def mincut(g=Graph, average=False):
    return g.mincut().value


def edge_betweenness(g=Graph, average=False, weights=[]):
    if (average):
        return mean(g.edge_betweenness(directed=True, weights=weights))
    if len(weights) > 0:
        return g.edge_betweenness(directed=True, weights=weights)

    return g.edge_betweenness(directed=True)


def clustering_coefficient(g=Graph, average=False):
    if (average):
        return mean(g.transitivity_undirected(mode='zero'))
    return g.transitivity_undirected(mode='zero')


def pagerank(g=Graph, average=False,  weights=[]):
    if (average):
        return mean(g.pagerank(directed=True, weights=weights))
    if len(weights) > 0:
        return g.pagerank(directed=True, weights=weights)

    return g.pagerank(directed=True)


def centralization(n, d=1, metrics=[], metric_name=''):
    denominator_factor = None

    if not metric_name:
        return np.nan

    if metric_name == 'degree':
        denominator_factor = d*(n-1)*(n-2)
    elif metric_name == 'betweenness':
        denominator_factor = (n - 1)**2 * (n - 2)
    elif metric_name == 'harmonic':
        denominator_factor = d*(n -3)/2
    elif metric_name == 'closeness':
        denominator_factor = (d*(n-1)*(n-2))/(2*n - 3)

    if not denominator_factor:
        return np.nan


    metrics = list(filter(lambda m: m > 0, metrics))

    if len(metrics) == 0:
        return 0
   
    max_node = np.max(metrics)
   
    numerator = (max_node - np.array(metrics))
    total_sum = 0
    for v in metrics:
        diff = (max_node - v)
        # print(diff)
        total_sum += diff

    sum_num = np.sum(numerator)

    result = sum_num / denominator_factor

    return result


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

    d = np.max(g.count_multiple())

    data = [
        g.vcount(),
        g.ecount(),
        # np.sum(g.degree(mode='in')),
        # np.sum(g.degree(mode='out')),
        g.maxdegree(mode='in', loops=True),
        g.maxdegree(mode='out', loops=True),
        mean_degree(g),
        # np.std(g.degree(mode='all')),
        get_avg_in_out_degree(g),
        get_avg_weighted_in_out_degree(g, field_name='duree'),
        get_avg_in_out_disbalance(g),
        get_unique_edges_vs_total(g),
        g.diameter(directed=True),
        g.radius(mode='all'),
        g.density(),
        g.average_path_length(directed=True),
        g.reciprocity(ignore_loops=False),
        mean(g.eccentricity()),
        weak_size,
        strong_size,
        power_law_alpha(g),
        global_clustering_coefficient(g),
        clustering_coefficient(g, average=True),
        centralization(n=n, d=d, metrics=degree(g, mode='in'), metric_name='degree'),
        centralization(n=n, d=d, metrics=betweenness(g), metric_name='betweenness'),
        centralization(n=n, d=d, metrics=closeness(g), metric_name='closeness'),
        centralization(n=n, d=d, metrics=harmonic(g), metric_name='harmonic'),
        pagerank(g, average=True, weights=weights),
        degree_assortativity(g),
        homophily_nominal(g, 'age'),
        homophily_nominal(g, 'revenu'),
        homophily_nominal(g, 'ville'),
        homophily_nominal(g, 'region'),
        homophily_nominal(g, 'arrondissement'),
        homophily_nominal(g, 'address'),
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
    if type(duree_str) == str:
        pz = duree_str.split(":")
        ret += float(pz[0])
        ret += float(pz[1]) / 60 * 100
    elif type(duree_str) == int:
        ret = duree_str

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

    result = 0
    try:
        result = ratio_sum / (len(g.vs) - nb_isolated)
    except Exception as e:
        print('WARNING: Divide by zero error')

    # print(f'get_avg_in_out_degree: {result}')
    return result


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
            try:
                ratio_sum += (weight_in / (weight_in + weight_out))
            except Exception as e:
                continue
    result = 0
    try:
        result = ratio_sum / (len(g.vs) - nb_isolated)
    except Exception as e:
        print('WARNING: Divide by zero error')

    # print(f'get_avg_weighted_in_out_degree: {result}')
    return result


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
            disbalance_sum += max(indeg / (indeg + outdeg),
                                  1 - indeg/(indeg + outdeg))

    result = 0
    try:
        result = disbalance_sum / (len(g.vs) - nb_isolated)
    except Exception as e:
        print('WARNING: Divide by zero error')

    # print(f'get_avg_in_out_disbalance: {result}')

    return result


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

    # all_edges.sort()
    # print(all_edges)

    result = 0
    try:
        result = len(unique_edges) / nb_edges
    except Exception as e:
        print('WARNING: Divide by zero error')

    return result


def perform_filter(g, start_date, window_date):
    filters = {
        'age': '',
        'adresse': '',
        'arrondissement': '',
        'ville': '',
        'genre': '',
        'revenu': '',
        'date': '',
        'duree': '',
        'service': '',
        'accorderie': '',
    }
    filters['date'] = f':{start_date.strftime("%Y-%m-%d")},{window_date.strftime("%Y-%m-%d")}'

    snapshot = perform_filter_on_graph(g, filters=filters)

    return snapshot



def graph_novelty(g, sn_size, start_date, end_date, weighted=False, subset='NODE', degree_mode='all'):
    if len(g.vs) == 0:
        return np.nan

    window_date = start_date + timedelta(sn_size)

    result = []
    total_sum = 0
    while window_date < end_date:
        if window_date - timedelta(sn_size) == start_date:
            window_date = window_date + timedelta(sn_size)
            continue


        cummulative_snapshots = perform_filter(
            g,  start_date, window_date - timedelta(sn_size))

        current_snapshot = perform_filter(
            g, window_date - timedelta(sn_size), window_date)

        dataframe_cummulative_snapshots = []
        dataframe_current_snapshot = []

        if subset == 'EDGE':
            dataframe_cummulative_snapshots = cummulative_snapshots.get_edgelist()
            dataframe_current_snapshot = current_snapshot.get_edgelist()
        elif subset == 'NODE':
            dataframe_cummulative_snapshots = cummulative_snapshots.get_vertex_dataframe()['id'].unique()
            dataframe_current_snapshot = current_snapshot.get_vertex_dataframe()['id'].unique()
        else:
            break


        ratio_diff = 0

        if len(dataframe_cummulative_snapshots) == 0 or len(dataframe_current_snapshot) == 0:
            window_date = window_date + timedelta(sn_size)
            result.append((window_date.strftime("%Y/%m/%d"), ratio_diff))

            continue
       
        # print('current snapshot', dataframe_current_snapshot)
        # print('cummulative snapshot', dataframe_cummulative_snapshots)

        diff = set(dataframe_current_snapshot) - (set(dataframe_cummulative_snapshots))
        # print('====')
        # print(len(dataframe_current_snapshot), len(dataframe_cummulative_snapshots), len(diff))
        # print('diff', diff)
        

        if weighted and subset == 'NODE':
            diff_g = current_snapshot.vs.select(id_in=diff)
            ratio_diff = np.sum(diff_g.degree(mode=degree_mode)) / np.sum(current_snapshot.degree(mode=degree_mode))
        else:
             ratio_diff = len(diff) / len(dataframe_current_snapshot)

        result.append((window_date.strftime("%Y/%m/%d"), ratio_diff))

        total_sum = total_sum + ratio_diff

        window_date = window_date + timedelta(sn_size)

    norm = (end_date - start_date)/sn_size
   
    average = (1/norm.days)*total_sum

    return average, sn_size, result


def super_stars_count(g, threshold=.5, mode='all'):

    node_total = len(g.vs)
    degree_seq = g.degree(mode=mode)
    degree_seq.sort(reverse=True)

    degree_total = np.sum(degree_seq)
    
    node_sum = 0
    node_count = 0

    result = []

    for max_deg in degree_seq:
        node_sum += max_deg
        
        node_count+=1

        ratio = node_sum / degree_total
        result.append(max_deg/degree_total)
        # print(str(degree_total) +" - "+ str(node_sum) +" - "+ str(node_count) +" - "+ str(ratio) + " - "+ str(threshold))
        if ratio > threshold:
            break
    # if mode =='in':
    #     print('in super star count')
    #     print(degree_total)
    #     print(degree_seq)  
    # print(result)
    return node_total, node_count, result

def euclidean_distance(vector1, vector2):
    squared_diff = (vector1 - vector2) ** 2
    sum_squared_diff = np.sum(squared_diff)
    return np.sqrt(sum_squared_diff/len(vector1))

def node_attribute_variance(g):
    # age,genre,revenu,ville,region,arrondissement,adresse
    # detailservice
    result = []
    
    for node in g.vs:
        age_count = 0
        genre_count = 0
        revenu_count = 0
        ville_count = 0
        region_count = 0
        arrondissement_count = 0
        addresse_count = 0

        neighbors = g.neighbors(node)
        
        total_neighbor = len(neighbors)

        for neighbor_idx in neighbors:
            neighbor = g.vs[neighbor_idx]
            
            # print(node, neighbor)
            if neighbor['age'] == node['age']:
                # print('incrementing age')
                age_count+=1
            if neighbor['genre'] == node['genre']:
                # print('incrementing genre')
                genre_count+=1
            if neighbor['revenu'] == node['revenu']:
                # print('incrementing revenu')
                revenu_count+=1
            if neighbor['ville'] == node['ville']:
                # print('incrementing ville')
                ville_count+=1
            if neighbor['region'] == node['region']:
                # print('incrementing region')
                region_count+=1
            if neighbor['arrondissement'] == node['arrondissement']:
                # print('incrementing arrondissement')
                arrondissement_count+=1
            if neighbor['adresse'] == node['adresse']:
                # print('incrementing adresse')
                addresse_count+=1
        
        result.append([
            age_count/total_neighbor, 
            genre_count/total_neighbor, 
            revenu_count/total_neighbor, 
            ville_count/total_neighbor, 
            region_count/total_neighbor,
            arrondissement_count/total_neighbor,
            addresse_count/total_neighbor
        ])
    
    
    result = np.array(result)

    num_columns = result.shape[1]

    distances = []
    for i in range(num_columns):
        row= []
        for j in range(num_columns):
            distance = euclidean_distance(result[:, i], result[:,j])
            # print(f"Euclidean distance between rows {i} and {j}: {distance:.2f}")

            row.append(distance)
        distances.append(row)
    
    # construct an igraph 
    # print(np.array(distances)) 
    # Vertex attribute names
    attribute_names = ['age', 'genre', 'revenu', 'ville', 'region', 'arrondissement', 'adresse']

    gd = Graph()

    # Add vertices to the graph
    num_vertices = len(distances)
    gd.add_vertices(num_vertices)
    gd.vs['name'] = attribute_names
    # Add edges to the graph based on the adjacency matrix
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if distances[i][j] != 0:
                gd.add_edge(i, j, weight=distances[i][j])

    # Print the graph summary
    layout = gd.layout("fr")  # Fruchterman-Reingold layout
    # plot(gd, layout=layout, vertex_label=gd.vs["name"], vertex_label_color="black", edge_label=gd.es["weight"])
    # plot.show()
    # layout = g.layout("fr")  # Fruchterman-Reingold layout
    visual_style = {}
    visual_style["vertex_label"] = gd.vs["name"]  # Using "adresse" attribute for vertex labels
    visual_style["vertex_label_color"] = "black"
    visual_style["edge_label"] = gd.es["weight"]  # Using "weight" attribute for edge labels
    plot(gd, target="plot.png", layout=layout, **visual_style)