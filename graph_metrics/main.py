import os
from utils import get_start_and_end_date,  add_sheet_to_xlsx, create_xlsx_file, save_csv_file
from graph_loader import load_accorderie_network
from datetime import date
from snapshot_generator import create_snapshots
from metrics import compute_global_properties_on_graph, compute_graph_metrics
import argparse


output_dir = 'data/metrics/'


def main(span_days, folder_name="", g=None):
    folder_name = folder_name + '/'

    start_date, end_date = get_start_and_end_date(path=folder_name)

    if isinstance(start_date, date) == False or isinstance(end_date, date) == False or span_days < 1:
        raise "Incorrect date(s)"

    if folder_name == '' or folder_name is None:
        raise "Missing folder name"

    if g is None:
        g = load_accorderie_network(path=folder_name)

    folder_name = output_dir + folder_name + '/'

    # FIXME: make use of util function in graph_common
    # TODO: refactor utils in graph_metrics
    dir_exits = os.path.exists(folder_name)
    if not dir_exits:
        os.mkdir(folder_name)

    file_writer = create_xlsx_file(folder_name + '/graphs_metrics')

    global_metrics = compute_global_properties_on_graph(g=g)

    add_sheet_to_xlsx(file_writer=file_writer,
                      data=global_metrics, title='Global Metrics', index=True)

    # compute snapshots

    snapshots = create_snapshots(
        g, start_date=start_date, end_date=end_date, span_days=int(span_days))

    if len(snapshots) == 0:
        print('Could not create snapshots')
        return

    for snapshot in snapshots:
        sub_graph = snapshot['subgraph']

        metrics_row = compute_graph_metrics(g=sub_graph)

        add_sheet_to_xlsx(file_writer=file_writer,
                          data=metrics_row, title='from ' + snapshot['title'])

    # save file
    save_csv_file(file_writer=file_writer)

    return


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-s', '--span', default=30, type=int)
arg_parser.add_argument('-f', '--folder_name', default='analysis')

args = arg_parser.parse_args()

filters = args.__dict__
main(span_days=int(filters['span']), folder_name=filters['folder_name'])

'''
1. modifs on current metrics report 
    - remove average computation
2. on filter
    - filter snapshot by the filter query
    - compute average for each snapshot 
    - add average as last sheet
    - pass the filter file into plot for plotting
'''
