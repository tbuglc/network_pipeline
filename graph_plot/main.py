import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from matplotlib.backends.backend_pdf import PdfPages

# Set the directory path
dir_path = 'data/filters/Sherbrooke'

def gather_metrics(command):
    # Create an empty list to store data from all Excel files
    all_data = []

    # Loop through all files in the directory
    data = {
        "files": [],
        "metrics": {

        }
    }
    for filename in os.listdir(dir_path):
        if not filename.endswith('.xlsx'):
            break
        # Only read Excel files
        file_path = os.path.join(dir_path, filename)
        # Read the Excel file into a Pandas dataframe
        df = pd.read_excel(file_path, sheet_name=None)

        averages = df['Snapshot Average Metrics']
        for name, series in averages.items():
            if name == 'Unnamed: 0':
                continue
            if name not in data['metrics']:
                data['metrics'][name] = []
            data['metrics'][name].append(series)

        data['files'].append(filename)

            # Append the dataframe to the list

    return data
# print(data)

def plot_metrics_average():
    with PdfPages('data/plots/average_metric_plots.pdf') as pdf:
        # page per metrics

        for key, value in data['metrics'].items():
            idx = 0
            for v in value:
                plt.plot(v)
                # lg = data['files'][idx] + ' '+ key
                # print(lg)
                plt.legend(data['files'])
                idx += 1

            plt.title(key)
            pdf.savefig()
            plt.close()


# arg_parser = argparse.ArgumentParser()
#
# arg_parser.add_argument('-k', '--key')
# arg_parser.add_argument('-v', '--value')
# arg_parser.add_argument('-fd', '--folder_name')
#
# args = arg_parser.parse_args()
#
# filters = args.__dict__
#
plot_metrics_average()