from gensim.models import Word2Vec
from node2vec import Node2Vec
import networkx as nx
from structure import load_accorderie_network

def igraph_to_networkx(g):
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(g.vcount()))
    nx_graph.add_edges_from(g.get_edgelist())
    return nx_graph


g = load_accorderie_network('./data/sherbrooke/data/members.csv', './data/sherbrooke/data/transactions.csv')


node2vec = Node2Vec(igraph_to_networkx(g), dimensions=64, walk_length=30, num_walks=200, workers=4)


walks = node2vec.walks


model = node2vec.fit(window=10, min_count=1)


node_embeddings = model.wv







