import argparse
import os
from graph_loader import load_accorderie_network
from filters import perform_filter_on_graph
from utils import accorderies
from pathlib import Path


def main(filters={}):
    g = load_accorderie_network(filters['input'])

    g = perform_filter_on_graph(g, filters=filters)

    trx = g.get_edge_dataframe()
    members = g.get_vertex_dataframe()

    if trx is None and members is None:
        print('Not transactions and members found!')
        return

    file_dir = filters['output'] +'/' + accorderies[filters['accorderie'][0]]

    dir_exits = os.path.exists(file_dir)

    if (dir_exits == False):
        os.mkdir(Path(file_dir))

    members.to_csv(file_dir + '/members.csv', index=False)
    trx.to_csv(file_dir + '/transactions.csv', index=False)


arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('-i', '--input', required=True)
arg_parser.add_argument('-o', '--output', required=True)

arg_parser.add_argument('-f', '--filter', action="store_true")

arg_parser.add_argument('--age', action="append")
# help="--age: filter by age, i.e --age=12-23 for single value otherwise, chaining --age=12-23 --age=23-32 will return to a list ['12-23', '23-32']. Note that we perform OR operation on a list")
arg_parser.add_argument('--adresse', action="append")
# help="--adresse: filter by adresse, i.e --adresse=12 for single value otherwise, chaining --adresse=12 --adresse=23 will return to a list [12, 23]. Note that we perform OR operation on a list")
arg_parser.add_argument('--arrondissement', action="append")
# help="--arrondissement: filter by arrondissement, i.e --arrondissement=12 for single value otherwise, chaining --arrondissement=12 --arrondissement=23 will return to a list [12, 23]. Note that we perform OR operation on a list")
arg_parser.add_argument('--ville', action="append")
# help="--ville: filter by ville, i.e --ville=12 for single value otherwise, chaining --ville=12 --ville=23 will return to a list [12, 23]. Note that we perform OR operation on a list")
arg_parser.add_argument('--revenu', action="append")
# help="--revenu: filter by revenu, i.e --revenu=12 for single value otherwise, chaining --revenu=12 --revenu=23 will return to a list [12, 23]. Note that we perform OR operation on a list")
arg_parser.add_argument('--genre', action="append")
# help="--genre: filter by genre, i.e --genre=12 for single value otherwise, chaining --genre=12 --genre=23 will return to a list [12, 23]. Note that we perform OR operation on a list")


arg_parser.add_argument('--date', action="append")
# help="--date: filter by date, i.e --date=12 for single value otherwise, chaining --date=12 --date=23 will return to a list [12, 23]. Note that we perform OR operation on a list")
arg_parser.add_argument('--duree', action="append")
# help="--duree: filter by duree, i.e --duree=12 for single value otherwise, chaining --duree=12 --duree=23 will return to a list [12, 23]. Note that we perform OR operation on a list")
arg_parser.add_argument('--service', action="append")
# help="--service: filter by service, i.e --service=12 for single value otherwise, chaining --service=12 --service=23 will return to a list [12, 23]. Note that we perform OR operation on a list")

arg_parser.add_argument('--accorderie', action="append", required=True)
# help="--accorderie: filter by accorderie, i.e --accorderie=12 for single value otherwise, chaining --accorderie=12 --accorderie=23 will return to a list [12, 23]. Note that we perform OR operation on a list")


args = arg_parser.parse_args()

if (args is not None):
    filters = args.__dict__
    main(filters)
else:
    print("Missing filters")


