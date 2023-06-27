import os
import argparse
import pathlib
from multiprocessing import Pool
from datetime import datetime


def compute_metrics_on_graph(directories):
    # FIXME: uncomment this line below
    # walk_dir = os.path.join(input_dir, directories)
    list_files = []
    # FIXME: remove c
    c = 0
    # FIXME: TO REMOVE for loop below
    for walk_dir in directories:
        try:
            # list_files = os.listdir(walk_dir)
            list_files = os.listdir(os.path.join(input_dir, walk_dir))
            if ('metrics.xlsx' not in list_files  or os.stat(os.path.join(walk_dir,'metrics.xlsx')).st_size < 1024):
                # cmd = f"python graph_metrics/main.py -i={walk_dir} -o={os.path.join(walk_dir,'metrics.xlsx')} -s={30}"
                # os.system(cmd)
                print(walk_dir)
                c += 1
        except:
            continue

    
    # FIXME: REMOVE RETURN
    return c
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-i', '--input', required=True)


args = arg_parser.parse_args()

filters = args.__dict__

global input_dir
input_dir = filters['input']

if __name__ == '__main__':
    if input_dir:
        dirs = None
        try:
            print('INPUT='+input_dir)
            dirs = os.listdir(input_dir)
        except Exception as e:
            print('Incorrect path as input directory')

        if dirs:
            start_time = datetime.now()

#            with Pool(processes=None) as p:
#
#                p.map(compute_metrics_on_graph, dirs)
#                p.close()
#                p.join()
            
            counter = compute_metrics_on_graph(dirs)
            print('c = '+ str(counter))
            print('\n\n')

            end_time = datetime.now()
            print('START TIME: '+ start_time.strftime("%d/%m/%Y %H:%M:%S"))

            print('END TIME: '+ end_time.strftime("%d/%m/%Y %H:%M:%S"))

            print("--- %s seconds ---" % (end_time - start_time))
            print('Done!')
            print('\n\n')
    else:
        print('Input directory is required. i.e: -i = some/path')
