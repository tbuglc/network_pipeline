from igraph import Graph, statistics
import pandas as pd
import os
import numpy as np
import powerlaw as pl
from gensim.models import Word2Vec
from node2vec import Node2Vec
import networkx as nx
# from labeling import convert_folder_name_to_dict

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
    
    return l  


def giant_component(g):
    # print(g.vcount(), g.ecount())
    s_components = g.connected_components(mode='strong')
    w_components = g.connected_components(mode='weak')

    # print(s_components)
    # print('\n')
    # print(w_components)
    # # print(s_components)
    S_s = len(max(s_components, key=len)) / g.vcount() # giant strongly component
    S_w = len(max(w_components, key=len)) / g.vcount() # giant weakly component
    print(S_s, S_w)
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

def main(root_dir):
    root_dir = '.\\reduce_social'
    i =0 
    results = []
    for walk_dir, sub_dir, files in os.walk(root_dir):
        print(walk_dir, sub_dir, files)
        if len(sub_dir) == 0 and 'members.csv' in files and 'transactions.csv' in files:
            print('Calculating metrics of '+ str(i + 1))

            
            # folder_name = walk_dir.split('\\')[-1]

        # fld_to_dict = convert_folder_name_to_dict(folder_name)
        
        # target = [fld_to_dict['r'], fld_to_dict['s'], fld_to_dict['d']]
        
            g = load_accorderie_network(f'{walk_dir}\\members.csv', f'{walk_dir}\\transactions.csv')

            _, weakly = giant_component(g)
            l_cc, g_cc = clustering_coefficient(g)

            results.append(np.concatenate([[g.vcount(), g.ecount(), mean_degree(g), weakly, power_law_alpha(g), l_cc, g_cc, degree_assortativity(g)]]))
            
            i = i + 1

    df = pd.DataFrame(results)
    print(df.head(), df.shape)
    # df.columns = ['Vertices', 'Edges', 'Mean degree', 'Size of weakly connected component', 'Power Law Alpha', 'Local clustering coefficient', 'Global clustering coefficient', 'Homophily by degree']

    # # df.to_csv('structural_properties_acc_sherbrooke.csv')

# main()

def calcule_structure_properties(path):
    
    g = None 
    
    try:
        g =load_accorderie_network(f'{path}\\members.csv', f'{path}\\transactions.csv')
    except:
        pass
    if g == None or g.vcount() == 0:  return []

    strongly, weakly = giant_component(g)
    l_cc, g_cc = clustering_coefficient(g)
    age = homophily_nominal(g, 'age')
    revenu = homophily_nominal(g, 'revenu')
    ville = homophily_nominal(g, 'ville')
    region = homophily_nominal(g, 'region')
    arrondissement = homophily_nominal(g, 'arrondissement')
    addresse = homophily_nominal(g, 'adresse')
    average_length = small_world_mean_distance(g)
    result = np.concatenate([[g.vcount(), g.ecount(), mean_degree(g), average_length, weakly, strongly, power_law_alpha(g), l_cc, g_cc, degree_assortativity(g), age ,revenu ,ville ,region ,arrondissement ,addresse ]])
    
    return result