import os
import pandas as pd
import argparse
from graph_common.constants import metrics_columns
from graph_common.file_handler import add_sheet_to_xlsx, create_xlsx_file, save_csv_file
from graph_common.utils import create_folder_if_not_exist

'''
get accorderie name
read folder from metrics

create a dataframe
filter through each sheet
compute average
add computed average
save file
'''


def filter_report_file(filters):
    metrics = None
    try:
        metrics = pd.read_excel('data/metrics/' + filters['folder_name'] + '/graphs_metrics.xlsx', sheet_name=None)
    except ValueError:
        metrics = None

    averages = []
    indices = []

    dir_name = 'data/filters/' + filters['folder_name'] + '/'

    create_folder_if_not_exist(dir_name)

    file_writer = create_xlsx_file(
        dir_name + '/' + filters['key'] + '_' + filters['value'])

    add_sheet_to_xlsx(file_writer=file_writer,
                      data=metrics['Global Metrics'], title='Global Metrics', index=True)

    for key, sheet in metrics.items():

        if key == 'Global Metrics':
            continue

        filtered_sheet = sheet.loc[sheet[filters['key'].capitalize()] == filters['value']]

        if filtered_sheet.empty:
            # result[key] = filtered_sheet
            continue

        indices.append(key)
        add_sheet_to_xlsx(file_writer=file_writer,
                          data=filtered_sheet, title=key)

        avg_row = []
        for metric in metrics_columns:
            avg_row.append(filtered_sheet[metric].mean())

        averages.append(avg_row)

    pd_av_m = pd.DataFrame(data=averages, index=indices, columns=metrics_columns)

    add_sheet_to_xlsx(file_writer=file_writer, data=pd_av_m,
                      title='Snapshot Average Metrics', index=True)

    save_csv_file(file_writer=file_writer)


arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('-k', '--key')
arg_parser.add_argument('-v', '--value')
arg_parser.add_argument('-fd', '--folder_name')

args = arg_parser.parse_args()

filters = args.__dict__

filter_report_file(filters)
