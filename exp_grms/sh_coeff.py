import pandas as pd
from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from igraph import Graph
import os
import numpy as np
import argparse

accorderies = {
    2: "Québec",
    121: "Yukon",
    117: "La Manicouagan",
    86: "Trois-Rivières",
    88: "Mercier-Hochelaga-M.",
    92: "Shawinigan",
    113: "Montréal-Nord secteur Nord-Est",
    111: "Portneuf",
    104: "Montréal-Nord",
    108: "Rimouski-Neigette",
    109: "Sherbrooke",
    110: "La Matanie",
    112: "Granit",
    114: "Rosemont",
    115: "Longueuil",
    116: "Réseau Accorderie (du Qc)",
    118: "La Matapédia",
    119: "Grand Gaspé",
    120: "Granby et région",
}


pandas2ri.activate()

cran_repo_url = "https://mirror.csclub.uwaterloo.ca/CRAN/"
ro.r(f'options(repos="{cran_repo_url}")')

statnet = importr("statnet")

ergms = importr("ergm")

fit_function = """
perform_ergm_fitting <- function(members=NULL, transactions=NULL){
    new_members <- read.csv(members)
    transactions <- read.csv(transactions)
    
    g_network <- network(transactions, loops=TRUE, directed=TRUE, multiple=TRUE, vertex.attr = new_members)

    ergm_fit <- ergm(formula= g_network ~ 
        nodematch('degrees') + 
        nodematch('betweenness') + 
        nodematch('harmonic_centrality') + 
        nodematch('transitivity_undirected') + 
        nodematch('pagerank') + 
        nodematch('eccentricity'), 
    control = control.ergm())

    coefficients <- coef(ergm_fit)

    return(coefficients)
}
"""

ro.r(fit_function)


FILE_NAME = ''


def convert_folder_name_to_dict(s):
    d = {}

    for kv in s.split('_'):
        k, v = kv.split('-')
        if k == 'sd' and v == "rand":
            d['sp'] = 1
        if k == 'sd' and v == "exp":
            continue
        if k in d:
            continue

        if k not in d:
            try:
                d[k] = float(v)
            except:
                d[k] = v

    return d


def process_folder(walk_dir):
    print(walk_dir)
    folder_name = walk_dir.split('\\')[-1]

    # fld_to_dict = convert_folder_name_to_dict(folder_name)
    # target = [fld_to_dict['r'], fld_to_dict['sp'], fld_to_dict['d']]

    members = pd.read_csv(f'{walk_dir}/members.csv', encoding='latin-1')
    transactions = pd.read_csv(
        f'{walk_dir}/transactions.csv', encoding='latin-1')

    g = None

    try:
        g = Graph.DataFrame(transactions, directed=True, vertices=members)
    except ValueError as err:
        print("Failed to load graph")
        return

    degrees = g.degree()
    betweenness = g.betweenness(directed=True)
    harmonic_centrality = g.harmonic_centrality(mode="all")
    transitivity_undirected = g.transitivity_undirected(mode='zero')
    pagerank = g.pagerank(directed=True)
    eccentricity = g.eccentricity()

    members["degrees"] = degrees
    members["betweenness"] = betweenness
    members["harmonic_centrality"] = harmonic_centrality
    members["transitivity_undirected"] = transitivity_undirected
    members["pagerank"] = pagerank
    members["eccentricity"] = eccentricity

    members_temp_file = f'{walk_dir}/augmented_members.csv'

    members.to_csv(members_temp_file, encoding='latin-1')

    r_fit_function = ro.globalenv["perform_ergm_fitting"]

    coefficients = r_fit_function(
        members_temp_file, f'{walk_dir}/transactions.csv')

    with open(f'{FILE_NAME}.csv', 'a') as file:
        # Convert the numpy array to a string and write to the file
        file.write(','.join(map(str, np.concatenate(
            [coefficients, [folder_name], [accorderies[int(folder_name)]] ] ))) + '\n')


arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('-f', '--file_name', required=True)

arg_parser.add_argument('-c', '--chunk', required=True)

args = arg_parser.parse_args()

filters = args.__dict__

FILE_NAME = filters['file_name']

CHUNK = filters['chunk']

# Clear the shared result file before running the script
with open(f"{FILE_NAME}.csv", 'w') as file:
    pass


for walk_dir, sub_dirs, files in os.walk(f'{CHUNK}'):
    if len(sub_dirs) == 0:
        process_folder(walk_dir=walk_dir)


# # executor.map(process_folder, walk_dirs)
# for walk_dir, sub_dirs, files in os.walk("C:\\Users\\bugl2301\\Documents\\beluga\\test_report"):
#     if len(sub_dirs) == 0:
#         process_folder(walk_dir)

print('Done!')
