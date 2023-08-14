import os
import pandas as pd
import numpy as np
import concurrent.futures
import threading
import argparse


FILE_NAME = ''

def convert_folder_name_to_dict(s):
    d = {}
    for kv in s.split('_'):
        # print("kv" + kv)
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


# A shared lock for synchronization during file writing
file_write_lock = threading.Lock()

def process_folder(walk_dir):
    if 'metrics.xlsx' not in os.listdir(walk_dir):
        return 
 
    try:
        folder_name = walk_dir.split('/')[-1]
 
        fld_to_dict = convert_folder_name_to_dict(folder_name)
        target = [fld_to_dict['r'], fld_to_dict['sp'], fld_to_dict['d']]

        df = pd.read_excel(walk_dir+'/metrics.xlsx', sheet_name=None)
        result_df = df['Global Metrics'].loc[:, 'Value'].values.reshape(-1,)

        sn_features_df = df['Snapshot Average Metrics']
        r = sn_features_df.iloc[:, 1:].values.reshape(-1,)
        sn_global_metric = df['Snapshots Global Metrics'].iloc[:, 1:3].values.reshape(-1,)

        result_df = np.concatenate([result_df, sn_global_metric, r, target])

        # Acquire the lock before writing to the file
        with file_write_lock:
            with open(f'{FILE_NAME}.csv', 'a') as file:
                # Convert the numpy array to a string and write to the file
                file.write(','.join(map(str, result_df)) + '\n')

        # print(f'Processed: {walk_dir}')
    except Exception as e:
        print(e)
        return


arg_parser = argparse.ArgumentParser()


arg_parser.add_argument('-f', '--file_name', required=True)
arg_parser.add_argument('-c', '--chunk', required=True)


args = arg_parser.parse_args()

filters = args.__dict__


FILE_NAME = filters['file_name']
CHUNK = filters['chunk']

if __name__ == "__main__":
    # Clear the shared result file before running the script
    with open(f"{FILE_NAME}.csv", 'w') as file:
        pass

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Process each folder in parallel using threads
        
        walk_dirs = []
        for walk_dir, sub_dirs, files in os.walk(f'{CHUNK}'):
            if len(sub_dirs) == 0:
                walk_dirs.append(walk_dir)

        executor.map(process_folder, walk_dirs)

    print('Done!')
