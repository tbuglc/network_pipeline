from igraph import Graph, statistics
import pandas as pd
import os
import numpy as np
import powerlaw as pl
from gensim.models import Word2Vec
from node2vec import Node2Vec
import networkx as nx

def data_loader(vertex_path, edge_path):
    # TODO: Should consider loading as stream for better memory usage in case of large dataset
    users = pd.read_csv(vertex_path, encoding='latin-1')
    transactions = pd.read_csv(edge_path, encoding='latin-1')

    return users, transactions


def load_accorderie_network(vertex_path, edge_path):
    users, transactions = data_loader(vertex_path, edge_path)

    g = None

    try:
        g = Graph.DataFrame(transactions, directed=True, vertices=users)
    except ValueError as err:
        print("Failed to load graph")
        raise err

    return g


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
    s_components = g.clusters(mode='strong')
    w_components = g.clusters(mode='weak')
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


# C:\Users\bugl2301\Downloads\biais_no_social.zip\u-200_t-5000_rb-0_db-1_sp-1.5_sd-e
g = load_accorderie_network('./data/sherbrooke/data/members.csv', './data/sherbrooke/data/transactions.csv')

# weakly, _ = giant_component(g)
# l_cc, g_cc = clustering_coefficient(g)

# df = pd.DataFrame([[g.vcount(), g.ecount(), mean_degree(g), weakly, power_law_alpha(g), l_cc, g_cc, degree_assortativity(g)]])
# df.columns = ['Vertices', 'Edges', 'Mean degree', 'Size of weakly connected component', 'Power Law Alpha', 'Local clustering coefficient', 'Global clustering coefficient', 'Homophily by degree']

# df.to_csv('structure.csv')

def igraph_to_networkx(g):
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(g.vcount()))
    nx_graph.add_edges_from(g.get_edgelist())
    return nx_graph


node2vec = Node2Vec(igraph_to_networkx(g), dimensions=64, walk_length=30, num_walks=200, workers=4)


# Generate walks
walks = node2vec.walks

# Train the Word2Vec model
model = node2vec.fit(window=10, min_count=1)

# Get the node embeddings
node_embeddings = model.wv

# Get the embeddings for a specific node
node_embedding = node_embeddings

print(pd.DataFrame.from_dict(node_embedding,  orient='index').head())