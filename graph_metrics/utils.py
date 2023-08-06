import pandas as pd
import os
from dateutil import parser
from graph_loader import data_loader
from pathlib import Path

# FIXME: POTENTIAL DUPLICATE
metrics_columns = ['Degree', 'Betweenness', 'Closeness',
                   'Page Rank', 'Clustering Coefficient']

global_graph_indices = [
                        'Vertices', 
                        'Edges', 
                        'Max in-degree', 
                        'Max out-degree', 
                        'Mean degree', 
                        'Average in-out degree',
                        'Average weighted in-out degree',
                        'Average in-out disbalance',
                        'Unique edges',
                        'Diameter',
                        'Radius', 
                        'Density',
                        'Average path length', 
                        'Reciprocity', 
                        'Average eccentricity', 
                        'Weakly connected component', 
                        'Strongly connected component', 
                        'Power law alpha',
                        'Global clustering coefficient',
                        'Clustering coefficient',
                        'Degree centralization',
                        'Betweenness centralization',
                        'Closeness centralization',
                        'Harmonic distance centralization',
                        'Average page rank',
                        'Degree assortativity',
                        'Homophily by age',
                        'Homophily by revenu',
                        'Homophily by ville',
                        'Homophily by region',
                        'Homophily by arrondissement',
                        'Homophily by adresse',
                    ]
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


def filter_by_trans_date(edge, start, end):
    # y, m, d = edge['date'].split('-')
    trans_date = parser.parse(edge['date'])
    # print(trans_date)
    # date(int(y), int(m), int(d))
    if (start <= trans_date and trans_date <= end):
        return True
    return False


def get_start_and_end_date(input_dir=''):
    _, trx = data_loader(input_dir)

    return parser.parse(trx['date'].min()), parser.parse(trx['date'].max())


def create_folder_if_not_exist(input_dir):
    dir_exits = os.input_dir.exists(input_dir)
    if not dir_exits:
        os.mkdir(Path(input_dir))


def create_xlsx_file(file_name: str):
    if (file_name == ''):
        raise ValueError('Missing file name')

    return pd.ExcelWriter(
        file_name, engine='xlsxwriter')


def add_sheet_to_xlsx(file_writer=pd.ExcelWriter, data=pd.DataFrame, title='', index=False):
    data.to_excel(
        file_writer, sheet_name=title, index=index)


def save_csv_file(file_writer=pd.ExcelWriter):
    file_writer.close()


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

# DUPLICATED
def perform_filter_on_graph(g, filters):
    if (not filters):
        return None

    # vertices properties filter
    if (filters["age"]):
        print("Filtering by age, value= "+str(filters["age"]))
        g = g.induced_subgraph(g.vs.select(age_in=filters["age"]))

    if (filters["adresse"]):
        print("Filtering by adresse, value= "+str(filters["adresse"]))
        g = g.induced_subgraph(g.vs.select(adresse_in=filters["adresse"]))

    if (filters["arrondissement"]):
        print("Filtering by arrondissement, value= " +
              str(filters["arrondissement"]))
        g = g.induced_subgraph(g.vs.select(
            arrondissement_in=filters["arrondissement"]))

    if (filters["ville"]):
        print("Filtering by ville, value= "+str(filters["ville"]))
        g = g.induced_subgraph(g.vs.select(aville_in=filters["ville"]))

    if (filters["genre"]):
        print("Filtering by genre, value= "+str(filters["genre"]))
        g = g.induced_subgraph(g.vs.select(
            genre_in=[int(j) for j in filters["genre"]]))

    if (filters["revenu"]):
        rev = {''+filters["revenu"][0]: filters["revenu"][1]}
        print("Filtering by revenu, value= "+str(filters["revenu"]))
        g = g.induced_subgraph(g.vs(**rev))
    # edges properties
    if (filters["date"]):
        print("Filtering by date, value= "+str(filters["date"]))

        date_filter = filters['date']
        if date_filter[0] == '<':
            g = g.subgraph_edges(g.es.select(lambda e: False if e['date'] == '0000-00-00' else parser.parse(e['date']) <= parser.parse(date_filter[1:])))
        elif date_filter[0] == '>':
            g = g.subgraph_edges(g.es.select(lambda e: False if e['date'] == '0000-00-00' else parser.parse(e['date']) >= parser.parse(date_filter[1:])))
        elif date_filter[0] == ':':
            date_intervals = date_filter[1:].split(',')
            g = g.subgraph_edges(g.es.select(lambda e: False if e['date'] == '0000-00-00' else parser.parse(e['date']) >= parser.parse(date_intervals[0]) and parser.parse(e['date']) <= parser.parse(date_intervals[1])))
        else:
            g = g.subgraph_edges(g.es.select(date_in=[date_filter]))

    if (filters["duree"]):
        for i, d in enumerate(filters['duree']):
            splitted_d = d.split(':')
            if len(splitted_d[0]) < 2:
                filters['duree'][i] = '0' + filters['duree'][i] 
        
        print("Filtering by duree, value= "+str(filters["duree"]))
        g = g.subgraph_edges(g.es.select(duree_in=filters['duree']))
        

    if (filters["service"]):
        print("Filtering by service, value= "+str(filters["service"]))
        g = g.subgraph_edges(g.es.select(service_in=filters["service"]))

    if (filters["accorderie"]):
        filters['accorderie'] = [int(x) for x in filters['accorderie']]
        print("Filtering by accorderie, value= "+str(filters["accorderie"]))
        g = g.subgraph_edges(g.es.select(accorderie_in=filters["accorderie"]))

    return g
