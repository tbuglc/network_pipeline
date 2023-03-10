import os
from utils import get_start_and_end_date,  add_sheet_to_xlsx, create_xlsx_file, save_csv_file
from graph_loader import load_accorderie_network
from datetime import date
from snapshot_generator import create_snapshots
from metrics import compute_global_properties_on_graph, compute_graph_metrics, compute_average_metrics
import argparse
from pathlib import Path
from utils import create_folder_if_not_exist, parse_output_dir
import pandas as pd

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
    for snapshot in snapshots:
        sub_graph = snapshot['subgraph']

        metrics_row = compute_graph_metrics(g=sub_graph)

        average_metrics.append(compute_average_metrics(sub_graph))
        indices.append(snapshot['title'])

        add_sheet_to_xlsx(file_writer=file_writer,
                          data=metrics_row, title='from '+snapshot['title'])
    
    columns = ['Degree', 'Betweenness', 'Closeness',
               'Page Rank', 'Clustering Coefficient', 'Eccentricity', 'Mincut', 'Edge betweenness']

    pd_av_m = pd.DataFrame(
        data=average_metrics, index=indices, columns=columns)

    # add sheets into excel
    add_sheet_to_xlsx(file_writer=file_writer, data=pd_av_m,
                      title='Snapshot Average Metrics', index=True)

    print('Computed metrics on snapshots')
    # save file
    save_csv_file(file_writer=file_writer)

    return


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-i', '--input', required=True)
arg_parser.add_argument('-o', '--output', required=True)

arg_parser.add_argument('-s', '--span', default=30, type=int)


args = arg_parser.parse_args()

filters = args.__dict__
main(span_days=int(filters['span']), input_dir=filters['input'], output_dir=filters['output'])
