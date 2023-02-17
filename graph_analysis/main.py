import os
from pydoc import describe
import sys
from utils import get_start_and_end_date
from file_generator import add_sheet_to_xlsx, create_xlsx_file, save_csv_file, pdf_degree_distribution
from graph_loader import load_accorderie_network
from datetime import date, datetime
from snapshot_generator import create_snapshots
from metrics import compute_average_metrics, compute_global_properties_on_graph, compute_graph_metrics
from plots import plot_communities, plot_complete_graph, plot_graph_snapshots
import pandas as pd
import igraph as ig
import argparse
from dateutil import parser


def main(span_days, folder_name="", g=None):

    # if (len(sys.argv) > 1 and len(sys.argv) < 6):
    #     start_date = date(int(sys.argv[1].split(
    #         "/")[0]), int(sys.argv[1].split("/")[1]), int(sys.argv[1].split("/")[2]))
    #     end_date = date(int(sys.argv[2].split(
    #         "/")[0]), int(sys.argv[2].split("/")[1]), int(sys.argv[2].split("/")[2]))
    #     span_days = int(sys.argv[3])
    #     #folder_name = datetime.now().strftime('%f') + "_" + sys.argv[4] + "/"
    #     folder_name = (sys.argv[4] + "/").replace(" ", "")

    start_date, end_date = get_start_and_end_date()

    if (isinstance(start_date, date) == False or isinstance(end_date, date) == False or span_days < 1):
        raise "Incorrect date(s)"

    if (folder_name == '' or folder_name is None):
        raise "Missing folder name"
        # return

    # create experiment folder to house all artifacts

    # load graph complet

    if (g == None):
        g = load_accorderie_network()
        # return

    # folder_name = datetime.now().strftime('%f')+'/'
    # print(g)

    dir_exits = os.path.exists('./artifacts/')
    if (dir_exits == False):
        os.mkdir('./artifacts')

    if (os.path.exists('./artifacts/'+folder_name) is False):
        os.mkdir('./artifacts/'+folder_name)

    # create file
    file_writer = create_xlsx_file('./artifacts/'+folder_name+'graphs_metrics')
    # create layout
    layout = g.layout('kk')
    # compute metrics on complete graph
    global_metrics = compute_global_properties_on_graph(g=g)
    # add sheet into excel
    add_sheet_to_xlsx(file_writer=file_writer,
                      data=global_metrics, title='Global Metrics', index=True)
    # plot complete graph

    color_palette = {
        "length": len(set(g.es['service'])),
        "palette":  ig.RainbowPalette(n=len(set(g.es['service'])))
    }

    plot_complete_graph(g, layout=layout, title=folder_name +
                        'complete_grapth', color_palette=color_palette)
    # compute snapshots

    snapshots = create_snapshots(
        g, start_date=start_date, end_date=end_date, span_days=span_days)
    if (len(snapshots) == 0):
        print('Could not create snapshots')
        return
    # compute metrics on snapshots

    average_metrics = []
    indices = []
    for snapshot in snapshots:
        sub_graph = snapshot['subgraph']

        metrics_row = compute_graph_metrics(g=sub_graph)

        average_metrics.append(compute_average_metrics(sub_graph))
        indices.append(snapshot['title'])

        add_sheet_to_xlsx(file_writer=file_writer,
                          data=metrics_row, title='from '+snapshot['title'])

    # xa, ya = compute_degree_distribution(g=sub_graph)

    # plot_degree_distribution(
    #     xa=xa, ya=ya, folder_name=folder_name + 'snapshot_degree_distribution_', title=snapshot['title'])  # FIXME: remote space on date title
    # gxa, gya = compute_degree_distribution(g)

    # plot_degree_distribution(
    #     gxa, gya, folder_name=folder_name, title='graph_complete_degree_distribution')

    pdf_degree_distribution(
        complete_graph=g, snapshots=snapshots, folder_name=folder_name+'degree_distributions')

    columns = ['Degree', 'Betweenness', 'Closeness',
               'Page Rank', 'Clustering Coefficient', 'Eccentricity', 'Mincut', 'Edge betweenness']

    pd_av_m = pd.DataFrame(
        data=average_metrics, index=indices, columns=columns)

    # add sheets into excel
    add_sheet_to_xlsx(file_writer=file_writer, data=pd_av_m,
                      title='Snapshot Average Metrics', index=True)
    # plots

    plot_graph_snapshots(snapshots=snapshots, layout=layout,
                         title=folder_name+'snapshots', color_palette=color_palette)

    # save file
    save_csv_file(file_writer=file_writer)

    plot_communities(g, layout=layout, title=folder_name +
                     'clusters_')


# main(span_days=30, folder_name="accorderie-data/")


def perform_filter(filters):
    if (not filters):
        return None

    g = load_accorderie_network()

    # vertices properties filter

    if (filters["age"]):
        g = g.induced_subgraph(g.vs.select(age_in=filters["age"]))

    if (filters["adresse"]):
        g = g.induced_subgraph(g.vs.select(adresse_in=filters["adresse"]))

    if (filters["arrondissement"]):
        g = g.induced_subgraph(g.vs.select(
            arrondissement_in=filters["arrondissement"]))

    if (filters["ville"]):
        g = g.induced_subgraph(g.vs.select(aville_in=filters["ville"]))

    if (filters["genre"]):
        g = g.induced_subgraph(g.vs.select(
            genre_in=[int(j) for j in filters["genre"]]))

    if (filters["revenu"]):
        rev = {''+filters["revenu"][0]: filters["revenu"][1]}
        g = g.induced_subgraph(g.vs(**rev))

    # edges properties
    if (filters["date"]):

        rev = {}
        if (len(filters["date"]) == 1):
            rev['date_in'] = [parser.parse(d) for d in filters["date"][0]]
        else:
            rev['date_'+filters["date"][0]
                ] = [parser.parse(d) for d in filters["date"][1]]

        g = g.subgraph_edges(g.es(**rev))

    if (filters["duree"]):
        duree = {}
        if (len(filters["duree"]) == 1):
            duree['duree_in'] = [str(i) for i in filters["duree"][0]]
        else:
            duree['duree_'+filters["duree"][0]
                  ] = [str(i) for i in filters["duree"][1]]

        g = g.subgraph_edges(g.es.select(**duree))

    if (filters["service"]):
        g = g.subgraph_edges(g.es.select(service_in=filters["service"]))

    return g


def filter_graph(filters={}):

    g = perform_filter(filters=filters)

    trx = g.get_edge_dataframe()
    members = g.get_vertex_dataframe()

    # generate folder name
    # TODO: create folder if not exist, add timestamp
    folder_name = 'filtered_analysis/'

    if (g):
        # call main to calculate metrics
        main(span_days=30, folder_name=folder_name, g=g)

    folder_name = 'filtered_data/'
    t_file_writer = create_xlsx_file('./artifacts/'+folder_name+'transactions')
    add_sheet_to_xlsx(file_writer=t_file_writer, data=trx, title='trx')
    save_csv_file(file_writer=t_file_writer)

    m_file_writer = create_xlsx_file('./artifacts/'+folder_name+'members')
    add_sheet_to_xlsx(file_writer=m_file_writer, data=members, title='m')
    save_csv_file(file_writer=m_file_writer)

    return trx, members


arg_parser = argparse.ArgumentParser()
# list down all available properties for
# users or transactions

arg_parser.add_argument('-f', '--filter', action="store_true")

arg_parser.add_argument('--age', action="append",
                        help="--age: filter by age, i.e --age=12-23 for single value otherwise, chaining --age=12-23 --age=23-32 will return to a list ['12-23', '23-32']. Note that we perform OR operation on a list")
arg_parser.add_argument('--adresse', action="append",
                        help="--adresse: filter by adresse, i.e --adresse=12 for single value otherwise, chaining --adresse=12 --adresse=23 will return to a list [12, 23]. Note that we perform OR operation on a list")
arg_parser.add_argument('--arrondissement', action="append",
                        help="--arrondissement: filter by arrondissement, i.e --arrondissement=12 for single value otherwise, chaining --arrondissement=12 --arrondissement=23 will return to a list [12, 23]. Note that we perform OR operation on a list")
arg_parser.add_argument('--ville', action="append",
                        help="--ville: filter by ville, i.e --ville=12 for single value otherwise, chaining --ville=12 --ville=23 will return to a list [12, 23]. Note that we perform OR operation on a list")
arg_parser.add_argument('--revenu', action="append",
                        help="--revenu: filter by revenu, i.e --revenu=12 for single value otherwise, chaining --revenu=12 --revenu=23 will return to a list [12, 23]. Note that we perform OR operation on a list")
arg_parser.add_argument('--genre', action="append",
                        help="--genre: filter by genre, i.e --genre=12 for single value otherwise, chaining --genre=12 --genre=23 will return to a list [12, 23]. Note that we perform OR operation on a list")


arg_parser.add_argument('--date', action="append",
                        help="--date: filter by date, i.e --date=12 for single value otherwise, chaining --date=12 --date=23 will return to a list [12, 23]. Note that we perform OR operation on a list")
arg_parser.add_argument('--duree', action="append",
                        help="--duree: filter by duree, i.e --duree=12 for single value otherwise, chaining --duree=12 --duree=23 will return to a list [12, 23]. Note that we perform OR operation on a list")
arg_parser.add_argument('--service', action="append",
                        help="--service: filter by service, i.e --service=12 for single value otherwise, chaining --service=12 --service=23 will return to a list [12, 23]. Note that we perform OR operation on a list")


args = arg_parser.parse_args()


if (args.filter is True):
    filters = args.__dict__
    filter_graph(filters)
else:
    main(span_days=30, folder_name='analysis/')
