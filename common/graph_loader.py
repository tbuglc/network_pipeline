from igraph import Graph
import pandas as pd
import os

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


def test_context():
    print('running from: ' + os.path.abspath())