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
    users = pd.read_csv(os.path.join(vertex_path), encoding='latin-1')
    transactions = pd.read_csv(os.path.join(edge_path), encoding='latin-1')

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
