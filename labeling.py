import os
import pandas as pd
import numpy as np

root_dir = 'D:\Compute Metrics\data_30\generated_graphs'

def convert_folder_name_to_dict(s):
    d = {}
    for kv in s.split('_'):
        k, v = kv.split('-')
        try:
            d[k] = float(v)
        except: 
            d[k] = v
            
    return d

def date_label(n):
    try:
        if n > 1:
            return 'd-growing'
        if n == 1:
            return 'd-random'
        if n < 1: return 'd-declining'
    except Exception as e:
        
        print("Error date: ",type(n))
    
def region_label(n):
    try:
        if n == 0: return 'r-random'
        if n > 0 and n < 0.6: return 'r-weak'
        if n >= 0.6: return 'r-strong'
    except Exception as e:
        print("Error region: ",type(n))
    
def sociability_label(n):
    if isinstance(n, (int, float, complex)): return 's-exp'
    if n == 'rand': return 1
    return 's-random'

def folder_to_label(fld):
    d = convert_folder_name_to_dict(fld)

    r = region_label(d['r'])
    dt = date_label(d['d'])
    s = sociability_label(d['s'])
    
    return f'{r}_{s}_{dt}'

def max_snapshot_count(curr, new_max):
    if new_max > curr:
        return new_max
    return curr    

i = 0  
max_snapshot = 1

dataset = pd.DataFrame()
for walk_dir, sub_dir, files in os.walk(root_dir):
    if len(sub_dir) == 0 and 'metrics.xlsx' in files and os.stat(walk_dir+'\\metrics.xlsx').st_size > 1024:        
        # 1. read global metrics 
        folder_name = walk_dir.split('\\')[-1]
        
        fld_to_dict = convert_folder_name_to_dict(folder_name)
        
        target = [fld_to_dict['r'], fld_to_dict['sp'], fld_to_dict['d']]
        # print('target: ', target)
        # print('\n')

        # print(walk_dir)
        
        df = pd.read_excel( walk_dir+'\\metrics.xlsx', sheet_name=None)
    
        result_df = df['Global Metrics'].loc[:, 'Value'].values.reshape(-1,)
        
        sn_features_df = df['Snapshot Average Metrics']

        r = sn_features_df.iloc[:,1:].values.reshape(-1,)
        # print(df['Snapshots Global Metrics'].iloc[:,1:3])
        sn_global_metric = df['Snapshots Global Metrics'].iloc[:,1:3].values.reshape(-1,)

        # print(sn_global_metric)
        # break
        # result_df = np.concatenate([result_df, sn_global_metric, r])
        result_df = np.concatenate([result_df, sn_global_metric, r, target])

        
        dataset = pd.concat([dataset, pd.DataFrame([result_df])], ignore_index=True, axis=0)
        
      
        max_snapshot = max_snapshot_count(max_snapshot, sn_features_df.shape[0])

        # break
        # break
        # print(dataset)
        # break
        # if(i==20):
        #     break     
        
        i = i+1
        print(i)

print('Done!')



print(max_snapshot)
# columns = []
# for s  in range(max_snapshot):
#     for k in ["Degree","Max in-degree","Max out-degree","Betweenness","Closeness","Harmonic distance","Page Rank","Average clustering coefficient","Global clustering coefficient","Edge betweenness","Vertices","Edges","Max in-degree","Max out-degree","Mean degree","Average in-out degree","Average weighted in-out degree","Average in-out disbalance","Unique edges","Diameter","Radius","Density","Average path length","Reciprocity","Average eccentricity","Weakly connected component","Strongly connected component","Power law alpha","Global clustering coefficient","Clustering coefficient","Degree centralization","Betweenness centralization","Closeness centralization","Harmonic distance centralization","Average page rank","Degree assortativity","Homophily by age","Homophily by revenu","Homophily by ville","Homophily by region","Homophily by arrondissement","Homophily by adresse"]
#         columns.append(f's{s}_{k}')
gbl_columns = [
"Vertices",
"Edges",
"Max in-degree",
"Max out-degree",
"Mean degree",
"Average in-out degree",
"Average weighted in-out degree",
"Average in-out disbalance",
"Unique edges",
"Diameter",
"Radius",
"Density",
"Average path length",
"Reciprocity",
"Average eccentricity",
"Weakly connected component",
"Strongly connected component",
"Power law alpha",
"Global clustering coefficient",
"Clustering coefficient",
"Degree centralization",
"Betweenness centralization",
"Closeness centralization",
"Harmonic distance centralization",
"Average page rank",
"Degree assortativity",
"Homophily by age",
"Homophily by revenu",
"Homophily by ville",
"Homophily by region",
"Homophily by arrondissement",
"Homophily by adresse"]

# gbl_columns =["Vertices", "Edges", "Max in-degree", "Max out-degree", "Mean degree", "Average in-out degree", "Average weighted in-out degree", "Average in-out disbalance", "Unique edges", "Diameter", "Radius", "Density", "Average path length", "Reciprocity", "Average eccentricity", "Weakly connected component", "Strongly connected component", "Power law alpha", "Global clustering coefficient", "Clustering coefficient", "Degree centralization", "Betweenness centralization", "Closeness centralization", "Harmonic distance centralization", "Average page rank", "Degree assortativity", "Homophily by age", "Homophily by revenu", "Homophily by ville", "Homophily by region", "Homophily by arrondissement", "Homophily by adresse"]

# gbl_columns = ['Vertices','Edges','Diameter', 'Radius', 'Density', 'Average path lenght', 'Reciprocity', 'Eccentricity', 'Clustering coefficient', 'Edge betweenness']

# all_columns = np.concatenate([gbl_columns, columns , ['target1', 'target2','target3']])
# all_columns = np.concatenate([gbl_columns, columns ])

# print(all_columns.shape, dataset.shape)

# dataset.columns = all_columns

dataset.to_csv('graph_120.csv')
