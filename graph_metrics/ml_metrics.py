from queue import Queue
import multiprocessing
import threading
import csv
import os
import argparse
from utils import load_accorderie_network
from metrics import compute_global_properties_on_snapshot
from igraph import Graph
from dateutil import parser
import numpy as np

start_date = "01/01/2006"
end_date = "12/31/2022"

output_file = 'metrics.csv'


def writer_thread(q: Queue, lock):
    while True:
        metrics = q.get()
        if metrics is None:  # Signal to stop the thread
            break

        print('PULLING FROM QUEUE AND WRITING')
        # print('METRICs FROM THE QUEUE: ', metrics)
        write_to_csv(lock, metrics)

        q.task_done()


def write_to_csv(lock, data):
    print('WRITING INTO FILE')
    with lock:
        print('ACQUIRED LOCK')
        with open(output_file, 'a', newline='') as file:
            writer = csv.writer(file)
            print('BEFORE DATA VALUE')

            writer.writerow(list(data))


def extract_bias_params(folder_name: str):
    name = (folder_name or '').split('/')[-1]

    result = []
    for bias in (name or '').split('_')[1:]:
        value = (bias or '').split('-')
        if len(value) > 2 and value[1] == '':
            value = '-'+value[-1]
        elif len(value) == 2:
            value = value[-1]
        else:
            value = np.nan

        # if value not in ['rand', 'exp']:
        if value == 'rand':
            result.append(float(0))
        elif value == 'exp':
            result.append(float(1))
        else:
            result.append(float(value))

    return result


def process_folders(folders, lock):
    q = Queue()
    writer = threading.Thread(target=writer_thread, args=(q, lock))
    writer.start()

    counter = 1
    total_folders = len(folders)
    print(f'FOLDER PER PROCESS LEN: {total_folders}')
    for folder in folders:
        # load network
        print(f'\n\n===== START PROCESSING: {folder} =====')
        g: Graph = load_accorderie_network(folder)
        # call metrics calculator
        if g is None:
            print('== SKIPING GRAPH ==')
            continue

        metrics = compute_global_properties_on_snapshot(
            g, 365, parser.parse(start_date),  parser.parse(end_date))
        print(f'===== COMPLETED PROCESSING: {folder} =====\n\n')
        metrics = np.append(np.array(metrics), np.array(
            extract_bias_params(folder_name=folder)))

        q.put(metrics.tolist())

        print(f'PROGRESS: {counter}/{total_folders}')
        counter += 1

    q.put(None)  # Signal the writer thread to stop
    writer.join()


def chunk_folders(folders, n):
    """Yield successive n-sized chunks from folders."""
    print('N: ', n)
    for i in range(0, len(folders), n):
        print(f'FROM FOLDER: {i} to {i+n}')
        yield folders[i:i + n]


def load_folders(base_path):
    # Get a list of directories
    directories = [os.path.join(base_path, d) for d in os.listdir(
        base_path) if os.path.isdir(os.path.join(base_path, d))]

    return directories


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-rf', '--root_folder', required=True)
arg_parser.add_argument('-fp', '--file_path', required=True)
arg_parser.add_argument('-p', '--processors', required=True)


args = arg_parser.parse_args()

filters = args.__dict__

output_file = filters['file_path']
if __name__ == '__main__':
    # Your list of 10,000 folder paths
    folder_paths = load_folders(base_path=filters['root_folder'])

    print('FOLDER PATH LEN: ', str(len(folder_paths)))

    num_processes = int(filters["processors"])  # Adjust as necessary
    folder_chunks = list(chunk_folders(
        folder_paths, len(folder_paths) // num_processes))

    with multiprocessing.Manager() as manager:
        lock = manager.Lock()

        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.starmap(process_folders, [(chunk, lock)
                                           for chunk in folder_chunks])
