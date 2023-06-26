import os
from file_generator import pdf_degree_distribution
from metrics import compute_degree_distribution
from plots import plot_degree_distribution
from utils import get_start_and_end_date,  add_sheet_to_xlsx, create_xlsx_file, save_csv_file
from graph_loader import load_accorderie_network
from datetime import date
from snapshot_generator import create_snapshots
from metrics import compute_global_properties_on_graph, compute_graph_metrics, compute_average_metrics, global_graph_properties
import argparse
from pathlib import Path
from utils import create_folder_if_not_exist, parse_output_dir, global_graph_indices
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main(span_days, input_dir, output_dir, g=None):

    start_date, end_date = get_start_and_end_date(input_dir=input_dir)

    if isinstance(start_date, date) == False or isinstance(end_date, date) == False or span_days < 1:
        raise "Incorrect date(s)"

    if output_dir == '' or output_dir is None:
        raise "Missing folder name"

    if g is None:
        g = load_accorderie_network(input_dir=input_dir)

    dest_dir, file_name = parse_output_dir(output_dir)

    create_folder_if_not_exist(dest_dir)

    if not file_name.endswith('.xlsx'):
        raise ValueError('Output dir should end with file.xlsx')

    file_writer = create_xlsx_file(output_dir)

    global_metrics = compute_global_properties_on_graph(g=g)

    add_sheet_to_xlsx(file_writer=file_writer,
                      data=global_metrics, title='Global Metrics', index=True)
    print('Computed global metrics')
    # compute snapshots
    print('Creating snapshots')
    snapshots = create_snapshots(
        g, start_date=start_date, end_date=end_date, span_days=int(span_days))
    print('Snapshots created')

    if len(snapshots) == 0:
        print('Could not create snapshots')
        return

    # for snapshot in snapshots:
    #     sub_graph = snapshot['subgraph']

    #     metrics_row = compute_graph_metrics(g=sub_graph)

    #     add_sheet_to_xlsx(file_writer=file_writer,
    #                       data=metrics_row, title='from ' + snapshot['title'])

    average_metrics = []
    indices = []
    global_metrics_on_snapshot = []
    for snapshot in snapshots:
        sub_graph = snapshot['subgraph']

        global_metrics_on_snapshot.append(global_graph_properties(sub_graph))

        metrics_row = compute_graph_metrics(g=sub_graph)

        average_metrics.append(compute_average_metrics(sub_graph))
        indices.append(snapshot['title'])

        add_sheet_to_xlsx(file_writer=file_writer,
                          data=metrics_row, title='from '+snapshot['title'])

    # gxa, gya = compute_degree_distribution(g)

    # plot_degree_distribution(
    #     gxa, gya, folder_name=dest_dir+'/graph_degree_distribution', title='Graph degree distribution')

    pdf_degree_distribution(
        complete_graph=g, snapshots=snapshots, folder_name=dest_dir+'/snapshots_degree_distribution')

    columns = [
            'Degree', 
            'Max in-degree', 
            'Max out-degree', 
            'Betweenness',
            'Closeness',
            'Harmonic distance',
            'Page Rank', 
            'Average clustering coefficient', 
            'Global clustering coefficient', 
            'Edge betweenness']

    pd_av_m = pd.DataFrame(
        data=average_metrics, index=indices, columns=columns + global_graph_indices)

    # add sheets into excel
    add_sheet_to_xlsx(file_writer=file_writer, data=pd_av_m,
                      title='Snapshot Average Metrics', index=True)

    sn_gb_m = pd.DataFrame(data=global_metrics_on_snapshot,
                           index=indices, columns=global_graph_indices)

    add_sheet_to_xlsx(file_writer=file_writer, data=sn_gb_m,
                      index=True, title="Snapshots Global Metrics")

    print('Computed metrics on snapshots')
    # save file
    save_csv_file(file_writer=file_writer)

    fig, ax = plt.subplots()

    # degree_seq = g.degree()

    xa, ya = zip(*[(int(left), count) for left, _, count in
                   g.degree_distribution().bins()])

    # plot the degree distribution
    ax.loglog(xa, ya)
    ax.set_xlabel('Degree')
    ax.set_ylabel('Count')
    ax.set_title('Degree Distribution')

    # FIXME: create a reusable block of code below
    path_dir = ''
    if '/' in output_dir:
        path_dir = '/'.join(output_dir.split('/')[0:-1])
    elif '\\' in output_dir:
        path_dir = '\\'.join(output_dir.split('\\')[0:-1])

    fig.savefig(Path(path_dir+'/degree_distribution.pdf'), dpi=300)
    plt.close()

    return


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-i', '--input', required=True)
arg_parser.add_argument('-o', '--output', required=True)

arg_parser.add_argument('-s', '--span', default=30, type=int)


args = arg_parser.parse_args()

filters = args.__dict__
main(span_days=int(filters['span']),
     input_dir=filters['input'], output_dir=filters['output'])
