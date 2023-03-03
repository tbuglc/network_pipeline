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

output_dir = 'data/metrics/'


def main(span_days, folder_name="", g=None):

    # if (len(sys.argv) > 1 and len(sys.argv) < 6):
    #     start_date = date(int(sys.argv[1].split(
    #         "/")[0]), int(sys.argv[1].split("/")[1]), int(sys.argv[1].split("/")[2]))
    #     end_date = date(int(sys.argv[2].split(
    #         "/")[0]), int(sys.argv[2].split("/")[1]), int(sys.argv[2].split("/")[2]))
    #     span_days = int(sys.argv[3])
    #     #folder_name = datetime.now().strftime('%f') + "_" + sys.argv[4] + "/"
    #     folder_name = (sys.argv[4] + "/").replace(" ", "")
    
    folder_name = folder_name + '/'

    start_date, end_date = get_start_and_end_date(path=folder_name)

    if (isinstance(start_date, date) == False or isinstance(end_date, date) == False or span_days < 1):
        raise "Incorrect date(s)"

    if (folder_name == '' or folder_name is None):
        raise "Missing folder name"
        # return

    # create experiment folder to house all output

    # load graph complet

    if (g == None):
        g = load_accorderie_network(path=folder_name)
        # return

    # folder_name = datetime.now().strftime('%f')+'/'
    # print(g)
    folder_name = output_dir+folder_name+'/'

    dir_exits = os.path.exists(folder_name)
    if (dir_exits == False):
        os.mkdir(folder_name)

    # if (os.path.exists(output_dir+folder_name) is False):
    #     os.mkdir(output_dir+folder_name)

    # create file
    file_writer = create_xlsx_file(folder_name+'/graphs_metrics')
    # create layout
    layout = g.layout('kk')
    # compute metrics on complete graph
    global_metrics = compute_global_properties_on_graph(g=g)
    # add sheet into excel
    add_sheet_to_xlsx(file_writer=file_writer,
                      data=global_metrics, title='Global Metrics', index=True)
    # plot complete graph

    # color_palette = {
    #     "length": len(set(g.es['service'])),
    #     "palette":  ig.RainbowPalette(n=len(set(g.es['service'])))
    # }

    # plot_complete_graph(g, layout=layout, title=folder_name +
    #                     'complete_grapth', color_palette=color_palette)
    # compute snapshots

    snapshots = create_snapshots(
        g, start_date=start_date, end_date=end_date, span_days=int(span_days))

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

    # pdf_degree_distribution(
    #     complete_graph=g, snapshots=snapshots, folder_name=folder_name+'degree_distributions')

    columns = ['Degree', 'Betweenness', 'Closeness',
               'Page Rank', 'Clustering Coefficient', 'Eccentricity', 'Mincut', 'Edge betweenness']

    pd_av_m = pd.DataFrame(
        data=average_metrics, index=indices, columns=columns)

    # add sheets into excel
    add_sheet_to_xlsx(file_writer=file_writer, data=pd_av_m,
                      title='Snapshot Average Metrics', index=True)
    # plots

    # plot_graph_snapshots(snapshots=snapshots, layout=layout,
    #                      title=folder_name+'snapshots', color_palette=color_palette)

    # save file
    save_csv_file(file_writer=file_writer)

    return

    # plot_communities(g, layout=layout, title=folder_name +
    #                  'clusters_')


# main(span_days=30, folder_name="accorderie-data/")




arg_parser = argparse.ArgumentParser()
# list down all available properties for
# users or transactions

arg_parser.add_argument('-s', '--span', default=30, type=int)
arg_parser.add_argument('-f', '--folder_name', default='analysis')

args = arg_parser.parse_args()



filters = args.__dict__
main(span_days=int(filters['span']), folder_name=filters['folder_name'])
