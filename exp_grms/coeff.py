import pandas as pd
from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri, conversion, default_converter
from rpy2.robjects.packages import importr
from igraph import Graph
import os
import numpy as np
import concurrent.futures
import threading
import argparse

import os
os.environ["R_ENABLE_JIT"] = "0"

file_write_lock = threading.Lock()

with conversion.localconverter(default_converter):
    # with file_write_lock:
    pandas2ri.activate()



cran_repo_url = "https://mirror.csclub.uwaterloo.ca/CRAN/"
with conversion.localconverter(default_converter):
    # with file_write_lock:
    # Set the repository using R's options function
    ro.r(f'options(repos="{cran_repo_url}")')
with conversion.localconverter(default_converter):
    # with file_write_lock:
    # Now import the statnet package
    statnet = importr("statnet")


# select a mirror for R packages
# statnet.chooseCRANmirror(ind=13)
with conversion.localconverter(default_converter):
    # with file_write_lock:
    ergms = importr("ergm")

# ergms.chooseCRANmirror(ind=13)

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


# file_write_lock = threading.Lock()

# @contextmanager
def process_folder(walk_dir):
    try:   
        folder_name = walk_dir.split('/')[-1]

        fld_to_dict = convert_folder_name_to_dict(folder_name)
        target = [fld_to_dict['r'], fld_to_dict['sp'], fld_to_dict['d']]

        # Step 1: Read members.csv and transactions.csv
        members = pd.read_csv(f'{walk_dir}/members.csv', encoding='latin-1')
        transactions = pd.read_csv(f'{walk_dir}/transactions.csv', encoding='latin-1')

        # Step 2: Construct igraph
        g = None
        try:
            g = Graph.DataFrame(transactions, directed=True, vertices=members)
        except ValueError as err:
            print("Failed to load graph")
            return
            # raise err
        # Step 3: Compute metrics
        degrees = g.degree()
        betweenness = g.betweenness(directed=True)
        closeness = g.closeness(mode="all")
        harmonic_centrality = g.harmonic_centrality(mode="all")
        # edge_betweenness = g.edge_betweenness(directed=True)
        transitivity_undirected = g.transitivity_undirected(mode='zero')
        pagerank = g.pagerank(directed=True)
        eccentricity = g.eccentricity()
        # Step 4: Add metrics into members dataframe

        members["degrees"] = degrees
        members["betweenness"] = betweenness
        members["closeness"] = closeness
        np.nan_to_num(closeness, nan=0)
        members["harmonic_centrality"] = harmonic_centrality
        # members["edge_betweenness"] = edge_betweenness
        members["transitivity_undirected"] = transitivity_undirected
        members["pagerank"] = pagerank
        members["eccentricity"] = eccentricity
        # Step 5: Save members dataframe into a temporary file
        members_temp_file = f'{walk_dir}/augmented_members.csv'

        members.to_csv(members_temp_file, encoding='latin-1')
        coefficients = []

        with conversion.localconverter(default_converter):
            # with file_write_lock:
                # Step 6: Construct R object
            ro.globalenv['members_temp_file'] = f'{members_temp_file}'
            ro.globalenv['transactions_file'] = f'{walk_dir}/transactions.csv'

            # Step 7: Load new_members and transactions in R
            ro.r('new_members <- read.csv(members_temp_file)')
            ro.r('transactions <- read.csv(transactions_file)')

            # Step 8: Construct R networks
            ro.r('network <- network(transactions, loops=TRUE, directed=TRUE, multiple=TRUE, vertex.attr = new_members)')
            ro.r('new_members')
            # Step 9: Compute ERGM
            ro.r("ergm_fit <- ergm(formula= network ~ nodematch('degrees') + nodematch('betweenness') + nodematch('closeness') + nodematch('harmonic_centrality') + nodematch('transitivity_undirected') + nodematch('pagerank') + nodematch('eccentricity'), control = control.ergm())")
        
            # Step 10: Extract coefficient
            # Your R code to extract coefficients here
            ro.r("coefficients <- coef(ergm_fit)")
        
            coefficients = ro.r['coefficients']

        # print(coefficients)

        # Acquire the lock before writing to the file
        with file_write_lock:
            with open(f'{FILE_NAME}.csv', 'a') as file:
                # Convert the numpy array to a string and write to the file
                file.write(','.join(map(str, np.concatenate([coefficients, target, [folder_name]]))) + '\n')
    except Exception as e:
        print(e)
        print(f'ERROR: process folder ==> folder name :{folder_name}')
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
