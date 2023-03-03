from igraph import Graph
import pandas as pd

input_dir = 'data/global_network/'

def data_loader():
    # TODO: Should consider loading as stream for better memory usage in case of large dataset
    users = pd.read_csv(input_dir+'members.csv', encoding='latin-1')
    transactions = pd.read_csv(input_dir+'transactions.csv', encoding='latin-1')

    return users, transactions


def load_accorderie_network():
    # TODO: Should consider loading as stream for better memory usage in case of large dataset
    users, transactions = data_loader()

    g = None

    try:
        g = Graph.DataFrame(transactions, directed=True, vertices=users)
    except ValueError as err:
        print("Failed to load graph")
        raise err

    return g