import pandas as pd
import os
from pathlib import Path

# FIXME: POTENTIAL DUPLICATE
metrics_columns = ['Degree', 'Betweenness', 'Closeness',
                   'Page Rank', 'Clustering Coefficient']


# FIXME: POTENTIAL DUPLICATE
accorderies = {
    2: "Québec",
    121: "Yukon",
    117: "La Manicouagan",
    86: "Trois-Rivières",
    88: "Mercier-Hochelaga-M.",
    92: "Shawinigan",
    113: "Montréal-Nord secteur Nord-Est",
    111: "Portneuf",
    104: "Montréal-Nord",
    108: "Rimouski-Neigette",
    109: "Sherbrooke",
    110: "La Matanie",
    112: "Granit",
    114: "Rosemont",
    115: "Longueuil",
    116: "Réseau Accorderie (du Qc)",
    118: "La Matapédia",
    119: "Grand Gaspé",
    120: "Granby et région",
}


'''
# FIXME: DUPLICATED FUNCTIONES
'''
def create_xlsx_file(file_name: str):
    if (file_name == ''):
        raise ValueError('Missing file name')

    return pd.ExcelWriter(
        file_name+'.xlsx', engine='xlsxwriter')


def add_sheet_to_xlsx(file_writer=pd.ExcelWriter, data=pd.DataFrame, title='', index=False):
    data.to_excel(
        file_writer, sheet_name=title, index=index)


def save_csv_file(file_writer=pd.ExcelWriter):
    file_writer.save()


def create_folder_if_not_exist(path):
    dir_exits = os.path.exists(path)
    if not dir_exits:
        os.mkdir(Path(path))

def parse_output_dir(path):
    if path == '':
        raise 'Path is required'
    file_name = ''
    if '/' in path:
        file_name = path.split('/')[-1]
    if '\\' in path:
        file_name = path.split('\\')[-1]
    
    if '.' not in file_name:
        raise ValueError('file missing extension')
    
    output_dir = path.split(file_name)[0]

    return output_dir, file_name
  
'''
# FIXME: END OF DUPLICATE
'''