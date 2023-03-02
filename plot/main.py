import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Set the directory path
dir_path = './input'

# Create an empty list to store data from all Excel files
all_data = []

# Loop through all files in the directory
data = {
    "files": [],
    "metrics": {

    }
}
for filename in os.listdir(dir_path):
    if filename.endswith('.xlsx'):  # Only read Excel files
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


# print(data)

def plot_metrics_average():
    with PdfPages('./output/average_metric_plots.pdf') as pdf:
        # page per metrics

        for key, value in data['metrics'].items():
            idx = 0
            for v in value:
                plt.plot(v, marker='o')
                # lg = data['files'][idx] + ' '+ key
                # print(lg)
                plt.legend(data['files'])
                idx += 1

            plt.title(key)
            pdf.savefig()
            plt.close()





plot_metrics_average()
# s1 = pd.Series([1, 3, 2, 4])
# s2 = pd.Series([1, 3, 2, 4])
#
# # Plot the series using Matplotlib
# plt.plot(s1, linestyle='dashdot')
# plt.plot(s2, linestyle='--', marker='o')
#
# plt.legend(data['files'])
# plt.title('My Series Plot')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')

# Display the plot
# plt.show()
# Concatenate all dataframes into a single dataframe
# merged_df = pd.concat(all_data, ignore_index=True

# Do something with the merged dataframe
