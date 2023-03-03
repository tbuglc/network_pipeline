from igraph import Graph
import pandas as pd

input_dir = 'data/accorderies'


def data_loader(path=''):

    path = input_dir + '/'+path

    # TODO: Should consider loading as stream for better memory usage in case of large dataset
    users = pd.read_csv(path + '/members.csv', encoding='latin-1')
    transactions = pd.read_csv(path + '/transactions.csv', encoding='latin-1')
    # print(users, transactions)
    return users, transactions


def load_accorderie_network(path=''):
    # TODO: Should consider loading as stream for better memory usage in case of large dataset
    users, transactions = data_loader(path)

    g = None

    try:
        g = Graph.DataFrame(transactions, directed=True, vertices=users)
    except ValueError as err:
        print("Failed to load graph")
        raise err

    return g
