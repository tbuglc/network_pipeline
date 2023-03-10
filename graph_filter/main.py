import argparse
import os
from graph_loader import load_accorderie_network
from filters import perform_filter_on_graph
from utils import accorderies
from pathlib import Path
from utils import create_folder_if_not_exist


def main(filters={}):
    g = load_accorderie_network(filters['input'])

    g = perform_filter_on_graph(g, filters=filters)

    trx = g.get_edge_dataframe()
    members = g.get_vertex_dataframe()

    if trx is None and members is None:
        print('Not transactions and members found!')
        return


    output_dir = filters['output']

    create_folder_if_not_exist(output_dir)

    members.to_csv(output_dir + '/members.csv', index=False)
    trx.to_csv(output_dir + '/transactions.csv', index=False)


arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('-i', '--input', required=True)
arg_parser.add_argument('-o', '--output', required=True)

# arg_parser.add_argument('-f', '--filter', action="store_true")

arg_parser.add_argument('--age', action="append")
arg_parser.add_argument('--adresse', action="append")
arg_parser.add_argument('--arrondissement', action="append")
arg_parser.add_argument('--ville', action="append")
arg_parser.add_argument('--revenu', action="append")
arg_parser.add_argument('--genre', action="append")


arg_parser.add_argument('--date', action="append")
arg_parser.add_argument('--duree', action="append")
arg_parser.add_argument('--service', action="append")

arg_parser.add_argument('--accorderie', action="append")


args = arg_parser.parse_args()

if (args is not None):
    filters = args.__dict__
    main(filters)
else:
    print("Missing filters")


