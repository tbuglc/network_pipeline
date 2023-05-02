import os
import pandas as pd

root_dir = 'C:\\Users\\bugl2301\\Documents\\generated_graphs'

def convert_str_to_dict(s):
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
    
    return 's-random'

def folder_to_label(fld):
    d = convert_str_to_dict(fld)

    r = region_label(d['r'])
    dt = date_label(d['d'])
    s = sociability_label(d['s'])
    
    return f'{r}_{s}_{dt}'
    
i = 0  
dataset = pd.DataFrame()
for walk_dir, sub_dir, files in os.walk(root_dir):
    if len(sub_dir) == 0 and 'metrics.xlsx' in files and os.stat(walk_dir+'\\metrics.xlsx').st_size > 1024:        
        # 1. read global metrics 
        folder_name = walk_dir.split('\\')[-1]
        
        target = folder_to_label(folder_name)

        df = pd.read_excel( walk_dir+'\\metrics.xlsx', sheet_name=None)
    
        df = df['Global Metrics'].T
        df['target'] = ['target',target]
        

        
        dataset = pd.concat([dataset, df.iloc[1:,:]], ignore_index=True, axis=0)
        
        # print(dataset)
        # break
        # if(i==50):
        #     break     
        
        i = i+1
        print(i)

print('Done!')

dataset.columns = ['Vertices','Edges','Diameter', 'Radius', 'Density', 'Average path lenght', 'Reciprocity', 'Eccentricity', 'Clustering coefficient', 'Edge betweenness', 'target']

dataset.to_csv('dataset.csv')