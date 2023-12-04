from graph_common.graph_loader import load_accorderie_network
from graph_metrics.utils import global_graph_indices
from graph_metrics.metrics import global_graph_properties
import os
import sys
import pandas as pd
import numpy as np
# from structure import calcule_structure_properties

import sys
import os
sys.path.append(os.curdir)


def create_folder_if_not_exist(path):
    dir_exits = os.path.exists(path)
    if not dir_exits:
        os.mkdir(path)


file_acc = pd.read_csv('data/raw/accorderie.csv', encoding='latin-1')


def filter_all_accorderies():

    accorderies = file_acc['NoAccorderie'].tolist()

    # print(accorderies)

    for acc in accorderies:
        # create_folder_if_not_exist()
        cmd = f'python graph_filter/main.py -i="data/raw/parsed" -o="accorderies/{acc}" --accorderie_edge={acc} --accorderie_node={acc}'
        print(cmd)
        os.system(cmd)


def convert_tabular_df_to_dict():
    r = file_acc.get(['NoAccorderie', 'Nom']).to_dict('tight')
    result = {}
    for d in r['data']:
        result[d[0]] = d[1]

    return result


filter_all_accorderies()

# def calculate_structural_properties():
#     results = []
#     accorderies = convert_tabular_df_to_dict()
#     for p,  f, h in os.walk('data\\accorderies'):

#         if len(h) == 0:
#             continue
#         # print(p,f,h)
#         # try:
#         id = p.split('\\')[-1]

#         try:
#             g =load_accorderie_network(f'{p}\\members.csv', f'{p}\\transactions.csv')
#         except Exception as e:
#             continue

#         r = global_graph_properties(g)

#         results.append(np.concatenate([[id, accorderies[int(id)]],r]))
#     # except Exception as e:
#         #     print('Error', e)
#         #     continue

#     df = pd.DataFrame(results)
#     # print(df.head(), df.shape)

#     # df.columns = ['AccorderieID','Name','Vertices', 'Edges', 'Average degree', 'Average path length', 'Weakly connected component(size)', 'Stongly connected component', 'Power law alpha', 'Local clustering coefficient', 'Global clustering coefficient', 'Homophily by degree', 'Homophily by age', 'Homophily by revenue', 'Homophily by city', 'Homophily by region', 'Homophily by (arrondissement)', 'Homophily by addresse']
#     df.columns = ['AccorderieID','Name'] + global_graph_indices
#     df.to_csv('all_acc.csv')

# calculate_structural_properties()
