import pandas as pd
import argparse
from utils import add_sheet_to_xlsx ,create_folder_if_not_exist, save_csv_file, create_xlsx_file, metrics_columns
from filters import perform_filter_on_dataframe


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

    file_name = ''
    for key in filters:
        print(key)
        if key == 'folder_name' or not filters[key]:
            continue

        if file_name == '':
            file_name += str(key + '_'.join(filters[key]))
        else:
            file_name += '_' + str(key + '_'.join(filters[key]))

    file_writer = create_xlsx_file(
        dir_name + '/' + file_name)

    add_sheet_to_xlsx(file_writer=file_writer,
                      data=metrics['Global Metrics'], title='Global Metrics', index=True)

    for key, sheet in metrics.items():

        if key == 'Global Metrics':
            continue

        # Filter here
        filtered_sheet = perform_filter_on_dataframe(sheet, filters)

        if filtered_sheet.empty:
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

arg_parser.add_argument('-r', '--revenu', action='append')
arg_parser.add_argument('-a', '--age', action='append')
arg_parser.add_argument('-v', '--ville', action='append')
arg_parser.add_argument('-re', '--region', action='append')
arg_parser.add_argument('-ar', '--arrondissement', action='append')
arg_parser.add_argument('-fd', '--folder_name')

args = arg_parser.parse_args()

filters = args.__dict__

filter_report_file(filters)
