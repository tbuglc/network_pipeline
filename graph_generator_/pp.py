import powerlaw
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import datetime
from dateutil import parser

# users = pd.read_csv('./members.csv')
# transactions = pd.read_csv('./transactions.csv')

# h = ig.Graph.DataFrame(transactions, directed=True, vertices=users)

# fig, axs = plt.subplots()
# ig.plot(h, target=axs)

# plt.show()

def generate_degree_sequence(n, k, gamma):
    pl = powerlaw.Power_Law(xmin=1, parameters=[gamma])
    degrees = pl.generate_random(k)
    if ig.is_graphical_degree_sequence(degrees):
        return degrees
    else:
        return generate_degree_sequence(n, k, gamma)


def generate_date(start_date, end_date):
    a = 2  # shape parameter
    rand_num = np.random.power(a)

    # start_date = datetime.date(2014, 1, 1)

    # end_date = datetime.date(2023, 12, 31)

    num_days = (end_date - start_date).days

    rand_date = start_date + datetime.timedelta(days=int(rand_num * num_days))

    return rand_date


def choose_regions():
    pass


def add_graph_properties(g, start_date, end_date):
    # vertex
    g.vs['id'] = [None]* g.vcount()
    g.vs['age'] = [None]* g.vcount()
    g.vs['genre'] = [None]* g.vcount()
    g.vs['revenu'] = [None]* g.vcount()
    g.vs['region'] = [None]* g.vcount()
    g.vs['ville'] = [None]* g.vcount()
    g.vs['arrondissement'] = [None]* g.vcount()
    g.vs['accorderie'] = [None]* g.vcount()
    g.vs['mapid'] = [r for r in range(g.vcount())]
    
    # edge
    g.es['duree'] = [None] * g.ecount()
    g.es['date'] = [generate_date(start_date, end_date)
                     for r in range(g.ecount())]
    g.es['service'] = [None] * g.ecount()
    g.es['detailservice'] = [None] * g.ecount()
    g.es['accorderie'] = [None] * g.ecount()
    g.es['id'] = [None] * g.ecount()

    return g


def main(filters):
    print(filters)
    degree_sequence = generate_degree_sequence(
        int(filters['vertices']), int(filters['edges']), gamma=float(filters['alpha']))

    g = ig.Graph.Degree_Sequence(degree_sequence)
    start_date = parser.parse(filters['start_date'])
    end_date = parser.parse(filters['end_date'])

    g = add_graph_properties(g, datetime.date(start_date.year, start_date.month,
                             start_date.day), datetime.date(end_date.year, end_date.month, end_date.day))
    
    trx = g.get_edge_dataframe()
    members = g.get_vertex_dataframe()

    members.to_csv(filters['output'] + '/members.csv', index=False)
    trx.to_csv(filters['output'] + '/transactions.csv', index=False)


arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('-o', '--output', required=True)
arg_parser.add_argument('-v', '--vertices')
arg_parser.add_argument('-e', '--edges')
arg_parser.add_argument('-a', '--alpha')
arg_parser.add_argument('-sd', '--start_date')
arg_parser.add_argument('-ed', '--end_date')

args = arg_parser.parse_args()

filters = args.__dict__

main(filters)
