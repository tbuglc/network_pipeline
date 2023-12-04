
from numpy import mean
from igraph import Graph, plot, ADJ_DIRECTED
import pandas as pd
import numpy as np
import powerlaw as pl
from utils import global_graph_indices
from dateutil import parser
import sys
from collections import Counter
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
        denominator_factor = d*(n - 3)/2
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
        centralization(n=n, d=d, metrics=degree(
            g, mode='in'), metric_name='degree'),
        centralization(n=n, d=d, metrics=betweenness(g),
                       metric_name='betweenness'),
        centralization(n=n, d=d, metrics=closeness(g),
                       metric_name='closeness'),
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


def compute_blau_index(g, attr):
    attributes = [g.vs[node][attr] for node in range(
        len(g.vs)) if attr in g.vs[node].attributes()]
    print('\n\n\n\n\n\n\n')

    attribute_counts = Counter(attributes)
    print(attribute_counts)

    total = sum(attribute_counts.values())
    proportions = [count / total for count in attribute_counts.values()]
    print(proportions)
    print('\n\n\n\n\n\n\n')

    blaus_index = 1 - sum([p**2 for p in proportions])

    # result.append((attr, blaus_index))
    # return result
    return blaus_index


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
    all_ratio = []
    for v in g.vs:
        indeg = g.degree(v, mode='in')
        outdeg = g.degree(v, mode='out')

        if outdeg == 0 and indeg == 0:
            nb_isolated += 1
        else:
            ratio_sum += (indeg / (outdeg + indeg))
            all_ratio.append(indeg / (outdeg + indeg))
    result = 0

    try:
        print('ISOLATED: ', nb_isolated)
        result = ratio_sum / (len(g.vs) - nb_isolated)
    except Exception as e:
        print('WARNING: Divide by zero error')
    print(f'get_avg_in_out_degree: {result}')
    # FIXME:
    # std_in = np.std(g.degree(mode='in'))
    # std_out = np.std(g.degree(mode='out'))
    # print('in', g.degree(mode='in'), 'out', g.degree(mode='out'))
    # print('std_in: ', std_in)
    # print('std_out: ', std_out)
    # result = std_in / (std_in + std_out)
    print('RESULT IN-OUT ', result)
    return result, all_ratio


def disparity(g, mode='all'):
    result = []

    for v in g.vs:
        incident_edges = g.es[g.incident(v, mode=mode)]
    # Get the strength (sum of weights) of the links for the vertex
        strength = sum([duree_to_int(e['duree'])
                        for e in incident_edges])

        # Compute the disparity
        disparity = 0
        if strength > 0:
            disparity = sum([(duree_to_int(e['duree']) / strength) **
                            2 for e in incident_edges])

        result.append(disparity)

    return sum(result)/len(g.vs), result


# average duree_in / (duree_in + duree_out).  Less than 0.5 => outdeg bias, higher => indeg bias
def get_avg_weighted_in_out_degree(g, field_name='duree'):

    if len(g.vs) == 0:
        return -1

    ratio_sum = 0
    nb_isolated = 0
    total_weigh_in = []
    total_weigh_out = []
    for v in g.vs:
        weight_in = 0
        for e in g.es[g.incident(v, mode='in')]:
            weight_in += duree_to_int(e['duree'])

        weight_out = 0
        for e in g.es[g.incident(v, mode='out')]:
            weight_out += duree_to_int(e['duree'])

        total_weigh_in.append(weight_in)
        total_weigh_out.append(weight_out)

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

    print(f'get_avg_weighted_in_out_degree: {result}')
    # total_weigh_in = np.std(total_weigh_in)
    # total_weigh_out = np.std(total_weigh_out)

    # result = total_weigh_in / (total_weigh_in + total_weigh_out)
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
            disb_value = max(indeg / (indeg + outdeg),
                             1 - indeg/(indeg + outdeg))
            # print('disb_value', disb_value)
            disbalance_sum += disb_value

    result = 0
    try:
        print('LEN.G ', len(g.vs))
        print('ISOLATED ', nb_isolated)

        result = disbalance_sum / (len(g.vs) - nb_isolated)
    except Exception as e:
        print('WARNING: Divide by zero error')

    print(f'get_avg_in_out_disbalance: {result}')
    # print('DISBALANCE: ', result)
    return result


# ratio of unique edges / edges.  Under 1 => edges are repeated
def get_unique_edges_vs_total(g):

    if len(g.es) == 0:
        return -1

    nb_edges = len(g.es)
    print(f'number of edges: {nb_edges}')
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
    print(f'returned value: {result}')
    return result


def perform_filter(g, start_date, window_date, acc_id=None):
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
        'accorderie_edge': '',
    }
    if acc_id:
        filters["accorderie_edge"] = acc_id
        filters['accorderie_node'] = acc_id
    filters['date'] = f':{start_date.strftime("%Y-%m-%d")},{window_date.strftime("%Y-%m-%d")}'

    snapshot = perform_filter_on_graph(g, filters=filters)

    return snapshot


def growth_rate(g, sn_size, start_date, end_date, id, weighted=False, subset='NODE', degree_mode='all'):
    if len(g.vs) == 0:
        return np.nan

    window_date = start_date + timedelta(sn_size * 1)

    result = []
    raw_result = []

    # perform_filter_on_graph(g, filters=)
    # i.e:  2014
    # prev = 2013:2014
    # curr = 2014:2015
    while (window_date) < end_date:
        prev_snapshot = perform_filter(
            g,  window_date - timedelta(sn_size), window_date, acc_id=id)

        current_snapshot = perform_filter(
            g, window_date, window_date + timedelta(sn_size), acc_id=id)
        if subset == 'NODE':
            if len(prev_snapshot.vs) <= 0:
                raw_result.append(
                    (len(prev_snapshot.vs), len(current_snapshot.vs)))
                result.append(((window_date).strftime("%Y/%m/%d"), 0))

            else:
                rate = (len(current_snapshot.vs) - len(prev_snapshot.vs)) / \
                    len(prev_snapshot.vs)
                raw_result.append(
                    (len(prev_snapshot.vs), len(current_snapshot.vs)))
                result.append(((window_date).strftime("%Y/%m/%d"), rate))
        elif subset == 'EDGE':
            if len(prev_snapshot.es) <= 0:
                raw_result.append(
                    (len(prev_snapshot.es), len(current_snapshot.es)))
                result.append(((window_date).strftime("%Y/%m/%d"), 0))
            else:
                rate = (len(current_snapshot.es) - len(prev_snapshot.es)) / \
                    len(prev_snapshot.es)
                raw_result.append(
                    (len(prev_snapshot.es), len(current_snapshot.es)))
                result.append(((window_date).strftime("%Y/%m/%d"), rate))
        else:
            raise 'Subset not specified'

        window_date = window_date + timedelta(sn_size)

    return result, raw_result


def graph_novelty(g, sn_size, start_date, end_date, id, weighted=False, subset='NODE', degree_mode='all', density=False):
    if len(g.vs) == 0:
        return np.nan

    window_date = start_date + timedelta(sn_size * 2)

    result = []
    raw_result = []
    densities = []
    total_sum = 0
    while (window_date - timedelta(sn_size)) < end_date:
        # if window_date - timedelta(sn_size) == start_date:
        #     window_date = window_date + timedelta(sn_size)
        #     continue

        # start from start date up to beginning of current snapshot.
        # This exclude current snapshot in the cummulative
        print('cummulative snapshots')
        cummulative_snapshots = perform_filter(
            g,  start_date, window_date - timedelta(sn_size), acc_id=id)
        # current snapshot: start date = end of cumm, end date = cummulative end + snapshot size
        print('current snapshot')
        current_snapshot = perform_filter(
            g, window_date - timedelta(sn_size), window_date, acc_id=id)

        nb_interaction_current_snapshot = len(current_snapshot.es)
        nb_members_current_snapshot = len(current_snapshot.vs)
        if density:
            # print('nbitcs', nb_interaction_current_snapshot)
            # print('nbmcs', nb_members_current_snapshot)

            if nb_members_current_snapshot and nb_interaction_current_snapshot:
                densities.append(nb_interaction_current_snapshot /
                                 (nb_members_current_snapshot*(nb_members_current_snapshot - 1)))
            else:
                densities.append(0)

        dataframe_cummulative_snapshots = []
        dataframe_current_snapshot = []

        if subset == 'EDGE':
            dataframe_cummulative_snapshots = cummulative_snapshots.get_edgelist()
            dataframe_current_snapshot = current_snapshot.get_edgelist()
        elif subset == 'NODE':
            dataframe_cummulative_snapshots = cummulative_snapshots.get_vertex_dataframe()[
                'mapid']
            dataframe_current_snapshot = current_snapshot.get_vertex_dataframe()[
                'mapid']

            # dataframe_cummulative_snapshots = dataframe_cummulative_snapshots.loc[dataframe_cummulative_snapshots['accorderie'].astype(
            #     int) == id]["id"].unique()
            # dataframe_current_snapshot = dataframe_current_snapshot.loc[dataframe_current_snapshot['accorderie'].astype(
            #     int) == id]["id"].unique()
            # print("cumm", dataframe_cummulative_snapshots)
            # print("curr", dataframe_current_snapshot)
        else:
            break

        ratio_diff = 0
        diff = set()
        if len(dataframe_cummulative_snapshots) == 0 or len(dataframe_current_snapshot) == 0:
            result.append(((window_date - timedelta(sn_size)
                            ).strftime("%Y/%m/%d"), ratio_diff))
            raw_result.append(
                (len(dataframe_cummulative_snapshots), len(dataframe_current_snapshot), ((window_date - timedelta(sn_size)).strftime("%Y/%m/%d"))))

            window_date = window_date + timedelta(sn_size)

            continue

        # print('current snapshot', dataframe_current_snapshot)
        # print('cummulative snapshot', dataframe_cummulative_snapshots)

        diff = set(dataframe_current_snapshot) - \
            (set(dataframe_cummulative_snapshots))
        # print('====')
        # print(len(dataframe_current_snapshot), len(dataframe_cummulative_snapshots), len(diff))
        # print('diff', diff)

        if weighted and subset == 'NODE':
            # print('DIFF', diff)
            diff_g = current_snapshot.vs.select(mapid_in=diff)
            # print('filtered graph: ', diff_g)
            # print('DIFF VS CURR DEGREES: ', diff_g.degree(
            #     mode=degree_mode), current_snapshot.degree(mode=degree_mode))
            ratio_diff = np.sum(diff_g.degree(mode=degree_mode)) / \
                np.sum(current_snapshot.degree(mode=degree_mode))
            # print('ratio weighted: ', ratio_diff)
        else:
            ratio_diff = len(diff) / len(dataframe_current_snapshot)

        raw_result.append(
            (len(diff), len(dataframe_current_snapshot), ((window_date - timedelta(sn_size)).strftime("%Y/%m/%d"))))

        result.append(
            ((window_date - timedelta(sn_size)).strftime("%Y/%m/%d"), ratio_diff))

        total_sum = total_sum + ratio_diff

        window_date = window_date + timedelta(sn_size)

    norm = (end_date - start_date)/sn_size

    average = (1/norm.days)*total_sum

    return average, sn_size, result, raw_result, densities


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

        node_count += 1
        if node_sum == 0 and degree_total == 0:
            result.append(0)
            continue
        elif degree_total == 0:
            result.append(0)
            continue
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
    # result.append(threshold)
    return node_total, node_count, result, threshold


def euclidean_distance(vector1, vector2):
    squared_diff = (vector1 - vector2) ** 2
    sum_squared_diff = np.sum(squared_diff)
    return np.sqrt(sum_squared_diff/len(vector1))


def count_node_attribute_categories(nodes, atr, categories):
    # print('nodes: ', nodes)
    # print('atr: ', atr)
    # print('categories: ', categories)

    result = []
    for c in categories:
        count = 0
        for n in nodes:
            if n[atr] == c:
                count += 1
        result.append(count)

    return result


def construct_graph(distances, attribute_names):
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
    return gd


def transform_array_to_obj(prop, arr):
    result = {}
    # print('BEFORE TRANSFORMATION: ', arr)
    i = 0
    for v in arr:
        if v not in ['nan', 0, '0'] and ('' or prop).lower() != 'genre':
            result[v] = i
            i += 1
        elif ('' or prop).lower() == 'genre' and ('N' not in [c.upper() for c in str(v)]):
            result[v] = i
            i += 1
        else:
            # skip aberrant values
            pass
    # print('AFTER TRANSFORMATION: ', result)
    return result


def convert_matrix_to_graph(name, data, attribute_names):
    # p = pd.DataFrame(data)
    # p.columns = attribute_names

    # p.to_csv(f'./distances/{name}_raw.csv')
    # print('DATA MATRIX: ', data)

    matrx = []
    for row in data:
        row_sum = np.sum(row)
        if row_sum > 0:
            matrx.append(
                (np.array(row)/row_sum) * 100)
        else:
            matrx.append(row)

    # p = pd.DataFrame(matrx)
    # p.columns = attribute_names

    # p.to_csv(f'./distances/{name}_percent.csv')

    g = Graph(directed=True)

    # Add vertices to the graph
    num_vertices = len(matrx)
    g.add_vertices(num_vertices)
    g.vs['indices'] = [k for k in range(num_vertices)]
    g.vs['name'] = attribute_names
    # Add edges to the graph based on the adjacency matrix
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if matrx[i][j] != 0:
                g.add_edge(i, j, weight=matrx[i][j])

    return g


def normalize_matrix(matrix):
    for key in matrix:
        for r in range(np.array(matrix[key]).shape[0]):
            if sum(matrix[key][r]) > 0:
                matrix[key][r] = np.array(matrix[key][r])/sum(matrix[key][r])

    return matrix

    # min_value = matrix.min()
    # max_value = matrix.max()

    # # Normalize the matrix to [0, 1]
    # normalized_matrix = (matrix - min_value) / (max_value - min_value)
    # print('normalization matrix start ', matrix)
    # print('normalization matrix end ', matrix)
    # return normalized_matrix
    # print('row before scaling', matrix[key][r])
    # print('row after scaling', matrix[r])


def count_interactions(matrx, ob, source_prop, target_prop, direction='UNDIRECTED'):
    if ob.get(source_prop, None) == None or ob.get(target_prop, None) == None:
        return

    if direction == 'SOURCE_TARGET':
        matrx[ob[source_prop], ob[target_prop]] += 1
    elif direction == 'TARGET_SOURCE':
        matrx[ob[target_prop], ob[source_prop]] += 1
    elif direction == 'UNDIRECTED':
        matrx[ob[source_prop], ob[target_prop]] += 1
        matrx[ob[target_prop], ob[source_prop]] += 1
    else:
        return


def node_attribute_variance(g, accorderie_name=''):
    # print(ages)
    ages = transform_array_to_obj('age', np.unique(g.vs["age"]))
    # print(genres)
    genres = transform_array_to_obj("genre", np.unique(g.vs["genre"]))
    # print(revenu)
    revenus = transform_array_to_obj("revenu", np.unique(g.vs["revenu"]))

    ages_matrx = {
        "Source-Target": np.zeros((len(ages), len(ages))),
        "Target-Source": np.zeros((len(ages), len(ages))),
        "Undirected": np.zeros((len(ages), len(ages))),
    }
    genres_matrx = {
        "Source-Target": np.zeros((len(genres), len(genres))),
        "Target-Source": np.zeros((len(genres), len(genres))),
        "Undirected": np.zeros((len(genres), len(genres))),
    }
    revenus_matrx = {
        "Source-Target": np.zeros((len(revenus), len(revenus))),
        "Target-Source": np.zeros((len(revenus), len(revenus))),
        "Undirected": np.zeros((len(revenus), len(revenus))),
    }

    for e in g.es:
        u, v = g.vs[e.source], g.vs[e.target]
        # if (u['age'] == '96-150' or v['age'] == '96-150'):
        #     print('AGE 95-150')
        #     print('SOURCE', u)
        #     print('LINK', e)
        #     print('TARGET', v)

        # age: source target
        count_interactions(matrx=ages_matrx['Source-Target'], ob=ages, direction='SOURCE_TARGET', source_prop=str(
            u['age']), target_prop=str(v['age']))
        # age: target source
        count_interactions(matrx=ages_matrx['Target-Source'], ob=ages,  direction='TARGET_SOURCE', source_prop=str(
            u['age']), target_prop=str(v['age']), )
        # age: undirected
        count_interactions(matrx=ages_matrx['Undirected'], ob=ages, source_prop=str(
            u['age']), target_prop=str(v['age']))

        # genre: source target
        count_interactions(matrx=genres_matrx['Source-Target'], ob=genres, direction='SOURCE_TARGET',
                           source_prop=str(u['genre']), target_prop=str(v['genre']))
        # genre: target source
        count_interactions(matrx=genres_matrx['Target-Source'], ob=genres, direction='TARGET_SOURCE',
                           source_prop=str(u['genre']), target_prop=str(v['genre']))
        # genre: undirected
        count_interactions(matrx=genres_matrx['Undirected'], ob=genres,
                           source_prop=str(u['genre']), target_prop=str(v['genre']))

        # revenu: source target
        count_interactions(matrx=revenus_matrx['Source-Target'], ob=revenus, direction='SOURCE_TARGET',
                           source_prop=str(u['revenu']), target_prop=str(v['revenu']))
        # revenu: target source
        count_interactions(matrx=revenus_matrx['Target-Source'], ob=revenus, direction='TARGET_SOURCE',
                           source_prop=str(u['revenu']), target_prop=str(v['revenu']))
        # revenu: undirected
        count_interactions(matrx=revenus_matrx['Undirected'], ob=revenus,
                           source_prop=str(u['revenu']), target_prop=str(v['revenu']))

    print(ages)
    print('raw age matrix: \n', ages_matrx)
    # print('normalized age matrix: \n', normalize_matrix(ages_matrx))
    print(genres)
    print('raw genres matrix: \n', genres_matrx)
    # print('normalized genres matrix: \n', normalize_matrix(genres_matrx))
    print(revenus)
    print('raw revenu matrix: \n', revenus_matrx)
    # print('normalized revenu matrix: \n', normalize_matrix(revenus_matrx))

    return [
        {"data": ages_matrx, "title": "Ages", "labels": [
            k for (k, _) in ages.items()]},
        {"data": genres_matrx, "title": "Genres", "labels": [
            k for (k, _) in genres.items()]},
        {"data": revenus_matrx, "title": "Revenus", "labels": [
            k for (k, _) in revenus.items()]},
    ]

    # villes = transform_array_to_obj("ville", np.unique(g.vs["ville"]))
    # print(revenus)
    # regions = transform_array_to_obj("region", np.unique(g.vs["region"]))
    # print(regions)
    # try:
    #     addresses = transform_array_to_obj(
    #         "adresse", np.unique(g.vs["adresse"]))
    # except:
    #     addresses = transform_array_to_obj(
    #         "address", np.unique(g.vs["address"]))

    # arrondissements = transform_array_to_obj(
    # "arrondissement", np.unique(g.vs["arrondissement"]))
    # villes_matrx = np.zeros((len(villes), len(villes)))
    # regions_matrx = np.zeros((len(regions), len(regions)))
    # arrondissements_matrx = np.zeros(
    #     (len(arrondissements), len(arrondissements)))
    # addresses_matrx = np.zeros((len(addresses), len(addresses)))

    # {"data": normalize_matrix(villes_matrx), "title": "Villes", "labels": [
    #     k for (k, _) in villes.items()]},
    # {"data": normalize_matrix(regions_matrx), "title": "Regions", "labels": [
    #     k for (k, _) in regions.items()]},
    # {"data": normalize_matrix(arrondissements_matrx), "title": "Arrondissements", "labels": [
    #     k for (k, _) in arrondissements.items()]},
    # {"data": normalize_matrix(addresses_matrx), "title": "Adresses", "labels": [
    #     k for (k, _) in addresses.items()]},
    # return [
    #     {"ages": convert_matrix_to_graph(f"{accorderie_name}_ages", ages_matrx, [
    #                                      k for (_, k) in ages.items()])},
    #     {"genres": convert_matrix_to_graph(f"{accorderie_name}_genres", genres_matrx, [
    #                                        k for (_, k) in genres.items()])},
    #     {"revenus": convert_matrix_to_graph(f"{accorderie_name}_revenus", revenus_matrx, [
    #                                         k for (_, k) in revenus.items()])},
    #     {"villes": convert_matrix_to_graph(f"{accorderie_name}_villes", villes_matrx, [
    #                                        k for (_, k) in villes.items()])},
    #     {"regions": convert_matrix_to_graph(f"{accorderie_name}_regions", regions_matrx, [
    #                                         k for (_, k) in regions.items()])},
    #     {"arrondissements": convert_matrix_to_graph(f"{accorderie_name}_arrondissements", arrondissements_matrx, [
    #                                                 k for (_, k) in arrondissements.items()])},
    #     {"addresses": convert_matrix_to_graph(f"{accorderie_name}_addresses", addresses_matrx, [
    #                                           k for (_, k) in addresses.items()])},
    # ]


def node_attribute_variance_(g):
    ages = np.unique(g.vs["age"])
    # print("ages", ages)

    genres = np.unique(g.vs["genre"])
    # print("genres", genres)

    revenus = np.unique(g.vs["revenu"])
    # print("revenus", revenus)

    villes = np.unique(g.vs["ville"])
    # print("villes", villes)

    regions = np.unique(g.vs["region"])
    # print("regions", regions)

    arrondissements = np.unique(g.vs["arrondissement"])
    # print("arrondissements", arrondissements)

    addresses = np.unique(g.vs["adresse"])
    # print("addresses", addresses)

    ages_atr_count = []
    genres_atr_count = []
    revenus_atr_count = []
    villes_atr_count = []
    regions_atr_count = []
    arrondissements_atr_count = []
    addresses_atr_count = []

    for node in g.vs:
        neighbor_idxs = g.neighbors(node)
        neighbors = g.vs[neighbor_idxs]

        # print('NODE: ', node)
        # print('NEIGHBORS ', neighbors)
        nodes = np.concatenate([[node], neighbors])

        nd_nghb_atr_count = count_node_attribute_categories(nodes, 'age', ages)
        ages_atr_count.append(nd_nghb_atr_count)

        nd_nghb_atr_count = count_node_attribute_categories(
            nodes, 'genre', genres)
        genres_atr_count.append(nd_nghb_atr_count)

        nd_nghb_atr_count = count_node_attribute_categories(
            nodes, 'revenu', revenus)
        revenus_atr_count.append(nd_nghb_atr_count)

        nd_nghb_atr_count = count_node_attribute_categories(
            nodes, 'ville', villes)
        villes_atr_count.append(nd_nghb_atr_count)

        nd_nghb_atr_count = count_node_attribute_categories(
            nodes, 'region', regions)
        regions_atr_count.append(nd_nghb_atr_count)

        nd_nghb_atr_count = count_node_attribute_categories(
            nodes, 'arrondissement', arrondissements)
        arrondissements_atr_count.append(nd_nghb_atr_count)

        nd_nghb_atr_count = count_node_attribute_categories(
            nodes, 'adresse', addresses)
        addresses_atr_count.append(nd_nghb_atr_count)

    corr_ages_atr_count = np.dot(
        np.array(ages_atr_count).T, np.array(ages_atr_count))
    # print("corr_ages_atr_count: ", corr_ages_atr_count)
    print(corr_ages_atr_count.shape)
    corr_genres_atr_count = np.dot(
        np.array(genres_atr_count).T, np.array(genres_atr_count))
    # print("corr_genres_atr_count: ", corr_genres_atr_count)
    print(corr_genres_atr_count.shape)
    corr_revenus_atr_count = np.dot(
        np.array(revenus_atr_count).T, np.array(revenus_atr_count))
    # print("corr_revenus_atr_count: ", corr_revenus_atr_count)
    print(corr_revenus_atr_count.shape)
    corr_villes_atr_count = np.dot(
        np.array(villes_atr_count).T, np.array(villes_atr_count))
    # print("corr_villes_atr_count: ", corr_villes_atr_count)
    print(corr_villes_atr_count.shape)
    corr_regions_atr_count = np.dot(
        np.array(regions_atr_count).T, np.array(regions_atr_count))
    # print("corr_regions_atr_count: ", corr_regions_atr_count)
    print(corr_regions_atr_count.shape)
    corr_arrondissements_atr_count = np.dot(
        np.array(arrondissements_atr_count).T, np.array(arrondissements_atr_count))
    # print("corr_arrondissements_atr_count: ", corr_arrondissements_atr_cou)
    print(corr_arrondissements_atr_count.shape)
    corr_addresses_atr_count = np.dot(
        np.array(addresses_atr_count).T, np.array(addresses_atr_count))
    # print("corr_addresses_atr_count: ", corr_addresses_atr_count)
    print(corr_addresses_atr_count.shape)
    # g_corr_ages_atr_count = Graph.Adjacency(
    #     corr_ages_atr_count.tolist(), mode=ADJ_DIRECTED)
    # # print("g_corr_ages_atr_count summary: ", g_corr_ages_atr_count.summary)
    # g_corr_ages_atr_count.vs["name"] = ages
    # g_corr_ages_atr_count.es["weight"] = np.zeros(len(ages))

    # g_corr_genres_atr_count = Graph.Adjacency(
    #     corr_genres_atr_count.tolist(), mode=ADJ_DIRECTED)
    # # print("g_corr_genres_atr_count summary: ",
    #     #   g_corr_genres_atr_count.summary())
    # g_corr_genres_atr_count.vs["name"] = genres
    # g_corr_genres_atr_count.es["weight"] = np.zeros(len(genres))

    # g_corr_revenus_atr_count = Graph.Adjacency(
    #     corr_revenus_atr_count.tolist(), mode=ADJ_DIRECTED)
    # # print("g_corr_revenus_atr_count summary: ",
    #     #   g_corr_revenus_atr_count.summary())
    # g_corr_revenus_atr_count.vs["name"] = revenus
    # g_corr_revenus_atr_count.es["weight"] = np.zeros(len(revenus))

    # g_corr_villes_atr_count = Graph.Adjacency(
    #     corr_villes_atr_count.tolist(), mode=ADJ_DIRECTED)
    # # print("g_corr_villes_atr_count summary: ",
    #     #   g_corr_villes_atr_count.summary())
    # g_corr_villes_atr_count.vs["name"] = villes
    # g_corr_villes_atr_count.es["weight"] = np.zeros(len(villes))

    # g_corr_regions_atr_count = Graph.Adjacency(
    #     corr_regions_atr_count.tolist(), mode=ADJ_DIRECTED)
    # # print("g_corr_regions_atr_count summary: ",
    #     #   g_corr_regions_atr_count.summary())
    # g_corr_regions_atr_count.vs["name"] = regions
    # g_corr_regions_atr_count.es["weight"] = np.zeros(len(regions))

    # g_corr_arrondissements_atr_count = Graph.Adjacency(
    #     corr_arrondissements_atr_count.tolist(), mode=ADJ_DIRECTED)
    # # print("g_corr_arrondissements_atr_count summary: ",
    #     #   g_corr_arrondissements_atr_count.summary())
    # g_corr_arrondissements_atr_count.vs["name"] = arrondissements
    # g_corr_arrondissements_atr_count.es["weight"] = np.zeros(len(arrondissements))

    # g_corr_addresses_atr_count = Graph.Adjacency(
    #     corr_addresses_atr_count.tolist(), mode=ADJ_DIRECTED)
    # # print("g_corr_addresses_atr_count summary: ",
    #     #   g_corr_addresses_atr_count.summary())
    # g_corr_addresses_atr_count.vs["name"] = addresses
    # g_corr_addresses_atr_count.es["weight"] = np.zeros(len(addresses))

    # return [g_corr_ages_atr_count, g_corr_genres_atr_count, g_corr_revenus_atr_count, g_corr_villes_atr_count, g_corr_regions_atr_count, g_corr_arrondissements_atr_count, g_corr_addresses_atr_count]
    return None
