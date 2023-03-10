import pandas as pd
import argparse
from utils import add_sheet_to_xlsx ,create_folder_if_not_exist, save_csv_file, create_xlsx_file, metrics_columns
from filters import perform_filter_on_dataframe
from pathlib import Path
from utils import create_folder_if_not_exist, parse_output_dir


def filter_report_file(filters):
    metrics = None
    try:
        metrics = pd.read_excel(Path(filters['input']), sheet_name=None)
    except ValueError:
        metrics = None
        return

    averages = []
    indices = []

    output_dir = filters['output']

    dest_dir, file_name = parse_output_dir(output_dir)

    create_folder_if_not_exist(dest_dir)

    if file_name.endswith('.xlsx'):
        raise ValueError('Output dir should end with file.xlsx')
    
    file_writer = create_xlsx_file(
        output_dir)

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

arg_parser.add_argument('-i', '--input', required=True)
arg_parser.add_argument('-o', '--output', required=True)

arg_parser.add_argument('-r', '--revenu', action='append')
arg_parser.add_argument('-a', '--age', action='append')
arg_parser.add_argument('-v', '--ville', action='append')
arg_parser.add_argument('-re', '--region', action='append')
arg_parser.add_argument('-ar', '--arrondissement', action='append')


args = arg_parser.parse_args()

filters = args.__dict__

filter_report_file(filters)
