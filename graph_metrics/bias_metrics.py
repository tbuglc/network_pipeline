import argparse
import os
import sys
import json
from graph_loader import data_loader, load_accorderie_network
from snapshot_generator import create_snapshots
from utils import get_start_and_end_date,  add_sheet_to_xlsx, create_xlsx_file, save_csv_file, accorderies
from metrics import graph_novelty, super_stars_count, get_avg_in_out_degree, get_avg_in_out_disbalance, get_avg_weighted_in_out_degree, get_unique_edges_vs_total
# accept input folder
from collections import OrderedDict
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
from math import ceil
from matplotlib.backends.backend_pdf import PdfPages


def compute_bias_metrics_og(input_dir_path, net_ids=[], snapshot_size = 365, super_star_threshold=.5):
    '''
        net_ids: serves as a filter
    '''    

    # result = {
    #     "Snapshot Size": snapshot_size,
    # }
    result = OrderedDict()
    result["metadata"] = {}

    result["metadata"]["snapshot_size"] = snapshot_size,
    result["metadata"]["super_star_threshold"] = super_star_threshold,
    # result["metadata"]["start_date"] = start_date.strftime("%Y-%m-%d"),
    # result["metadata"]["end_date"] = end_date.strftime("%Y-%m-%d"),
    result["Node Novelty"] = {"plt": "plot", "data": []}
    result["Edge Novelty"] = {"plt": "plot", "data": []}
    result["Weighted Node Novelty"] = {"plt": "plot", "data": []}
    result["Global Metrics"] = {"plt": "bar", "data": []}
    result["Super Stars Sum"] = {"plt": "pie", "data": []}
    result["Super Stars In-deg"] = {"plt": "pie", "data": []}
    result["Super Stars Out-deg"] = {"plt": "pie", "data": []}


    folders = []
    if not input_dir_path and len(net_ids) > 0:
        folders = net_ids
    else:
        folders = os.listdir(input_dir_path)
    
    for fd in folders:

        if len(net_ids) > 0 and fd not in net_ids:
            continue

        print('\n\n\n\nFOLDER: '+ fd)
        accorderie_name  = accorderies[int(fd)]
    
        print("Name: " + accorderie_name)
        # result[accorderie_name] = {}

        try:
            start_date, end_date = get_start_and_end_date(input_dir=os.path.join(input_dir_path, fd))
            g = load_accorderie_network(os.path.join(input_dir_path, fd))
            
            # print(g.vs[0:10])

            _, sn_size, node_novelty = graph_novelty(g, sn_size=snapshot_size, start_date=start_date, end_date=end_date)
            print('\nnode')
            _, sn_size, edge_novelty = graph_novelty(g, sn_size=snapshot_size, start_date=start_date, end_date=end_date, subset='EDGE')
            print('\nedge')

            _, sn_size, weighted_node_novelty = graph_novelty(g, sn_size=snapshot_size, start_date=start_date, end_date=end_date, weighted=True)
            print('\nweighted')

            super_star_sum = super_stars_count(g, super_star_threshold, mode='all')
            super_star_in = super_stars_count(g, super_star_threshold, mode='in')
            super_star_out = super_stars_count(g, super_star_threshold, mode='out')
            print('super start')

            # result[accorderie_name]['Node Novelty'] = node_novelty
            # result[accorderie_name]['Edge Novelty'] = edge_novelty
            # result[accorderie_name]['Weighted Node Novelty'] = weighted_node_novelty
            # result[accorderie_name]['Super Stars'] = super_star

            # result.append(collection)

            result['Node Novelty']["data"].append(node_novelty)
            result['Edge Novelty']["data"].append(edge_novelty)
            result['Weighted Node Novelty']["data"].append(weighted_node_novelty)
            global_metrics =[get_avg_in_out_degree(g=g) ,get_avg_in_out_disbalance(g=g) ,get_avg_weighted_in_out_degree(g=g) ,get_unique_edges_vs_total(g=g)]
            result['Global Metrics']["data"].append(global_metrics)
            result['Super Stars Sum']["data"].append(super_star_sum)
            result['Super Stars In-deg']["data"].append(super_star_in)
            result['Super Stars Out-deg']["data"].append(super_star_out)

        except Exception as e:
            print(e)
            break
    return result

    #     # create snapshots
    #     snapshots = create_snapshots(
    #     g, start_date=start_date, end_date=end_date, span_days=int(span_days))

def compute_bias_metrics(input_dir_path, net_ids=[], snapshot_size = 365, super_star_threshold=.5):
    '''
        net_ids: serves as a filter
    '''    
    result = OrderedDict()
    result["first_page"] = {
        "plt": 'plot',
        "accorderies": [],
        "snapshot_size": snapshot_size,
        "titles": ["Node Novelty", "Edge Novelty", "Weighted Node Novelty"],
        "data": {
            "metadata": [],
            "Node Novelty": [],
            "Edge Novelty": [],
            "Weighted Node Novelty": []
        }
    }
    result["second_page"] = {
        "plt": 'bar',
        "accorderies": [],
        "snapshot_size": snapshot_size,
        "titles": ["Global Metrics"],
        "data": {
            "metadata": [],
            "Global Metrics": []
        }
    }
    result["third_page"] = {
        "plt": 'pie',
        "accorderies": [],
        "snapshot_size": snapshot_size,
        "titles": ["Super Stars Sum", "Super Stars In-deg", "Super Stars Out-deg"],
        "data": {
            "metadata": [],
            "Super Stars Sum": [],
            "Super Stars In-deg": [],
            "Super Stars Out-deg": [],
        }
    }


    folders = []
    if not input_dir_path and len(net_ids) > 0:
        folders = net_ids
    else:
        folders = os.listdir(input_dir_path)
    
    for fd in folders:

        if len(net_ids) > 0 and fd not in net_ids:
            continue

        print('\n\n\n\nFOLDER: '+ fd)
        accorderie_name  = accorderies[int(fd)]
        result["first_page"]["accorderies"].append(accorderie_name)
        result["second_page"]["accorderies"].append(accorderie_name)
        result["third_page"]["accorderies"].append(accorderie_name)
        # print("Name: " + accorderie_name)
        # result[accorderie_name] = {}

        try:
            start_date, end_date = get_start_and_end_date(input_dir=os.path.join(input_dir_path, fd))
            g = load_accorderie_network(os.path.join(input_dir_path, fd))
            
            result["first_page"]["data"]["metadata"].append({"start_date": start_date, "end_date": end_date })
            result["second_page"]["data"]["metadata"].append({"start_date": start_date, "end_date": end_date })
            result["third_page"]["data"]["metadata"].append({"start_date": start_date, "end_date": end_date })
            
            _, sn_size, node_novelty = graph_novelty(g, sn_size=snapshot_size, start_date=start_date, end_date=end_date)
            print('\nnode')
            _, sn_size, edge_novelty = graph_novelty(g, sn_size=snapshot_size, start_date=start_date, end_date=end_date, subset='EDGE')
            print('\nedge')

            _, sn_size, weighted_node_novelty = graph_novelty(g, sn_size=snapshot_size, start_date=start_date, end_date=end_date, weighted=True)
            print('\nweighted')

            super_star_sum = super_stars_count(g=g, threshold=super_star_threshold, mode='all')
            super_star_in = super_stars_count(g=g, threshold=super_star_threshold, mode='in')
            super_star_out = super_stars_count(g=g, threshold=super_star_threshold, mode='out')
            print('super start')

            
            global_metrics =[get_avg_in_out_degree(g=g) ,get_avg_in_out_disbalance(g=g) ,get_avg_weighted_in_out_degree(g=g) ,get_unique_edges_vs_total(g=g)]
            
            result["first_page"]["data"]["Node Novelty"].append(node_novelty)
            result["first_page"]["data"]["Edge Novelty"].append(edge_novelty)
            result["first_page"]["data"]["Weighted Node Novelty"].append(weighted_node_novelty)
            
            result["second_page"]["data"]["Global Metrics"].append(global_metrics)
            
            result["third_page"]["data"]["Super Stars Sum"].append(super_star_sum)
            result["third_page"]["data"]["Super Stars In-deg"].append(super_star_in)
            result["third_page"]["data"]["Super Stars Out-deg"].append(super_star_out)
            

        except Exception as e:
            print(e)
            break
    return result



def bias_report_og(metrics_data):

    metadata = metrics_data['metadata']
    x_value = ceil((len(metrics_data.keys()) + len(metrics_data['Super Stars']['data']) - 1)/2)

    fig, ax = plt.subplots(x_value, 2,  figsize=(12, 8))

    row = 0
    col = 0
    
    for key in metrics_data:
        if key == 'metadata' or key == 'Global Metrics':
            continue
        
                # row+=1
        if col == 2:
            col = 0
            row+=1

        data = metrics_data[key]['data']
        plt_key = metrics_data[key]['plt']
        Y = []
        
        # mod = row % 2
        target = ax[row,col]

        if plt_key == 'pie':
            for i,d in enumerate(data):
                target = ax[row + i , ( i + col )%2]

                target.pie(d)
               
            # row+=1                  
        else: 
            for d in data:
                if plt_key == 'bar':
                    Y = np.arange(0, 1, .1)
                    target.bar(d, Y)
                else:
                    target.plot(d)
            target.set_title(key)
            target.set_ylabel('Proportion')
            target.set_xlabel('Days')
        
        col+=1

    global_metrics = metrics_data['Global Metrics']['data']
    # print(x_value)
    # plt.subplot(2,1,2)
    width = 0.5
    for idx, gm in enumerate(global_metrics):
        X = np.arange(len(gm)) + width
        if idx == 0:
            X = X - width
        
        plt.bar(X + width, gm)
        
    plt.savefig('./bias_metrics.pdf', dpi=300)
    plt.close()

def bias_report(metrics_data):
     
     with PdfPages('bias_report.pdf') as pdf:
        for key in metrics_data:
            plt_key = metrics_data[key]["plt"]

            data = metrics_data[key]["data"]
            titles = metrics_data[key]["titles"]
            accorderies = metrics_data[key]["accorderies"]

            idx = None
            if plt_key == 'plot':
                row = ceil((len(data) - 1.01)/ 2)
                # print(data)
                col = ceil(len(metrics_data[key]["titles"])/2)
                print(row, col)
                fig, axes = plt.subplots(row, col)

                axes = axes.flatten()
                idx = 0
                for kd in data:
                    if kd == 'metadata':
                        continue
                    axes[idx].set_title(titles[idx])
                    for d in data[kd]:
                        # print(kd)
                        # print(data[kd])
                        axes[idx].plot(d)
                    axes[idx].set_ylabel('Proportion')
                    axes[idx].set_xlabel('Days')
                    idx+=1

            if plt_key == 'bar':
                row = ceil(len(data) - 1.01/ 2)
                
                col = len(metrics_data[key]["titles"])
            
                fig, axes = plt.subplots(row, col)

                axes = axes.flatten()
                idx = 0
                width = 0.5
                for kd in data:

                   
                    if kd == 'metadata':
                        continue

                    axes[idx].set_title(titles[idx])
                    for i,d in enumerate(data[kd]):
                        X = np.arange(len(d))
                        axes[idx].bar(X + (i - 1) * width ,d, width=width)
                    
                    axes[idx].set_ylabel('Proportion')
                    axes[idx].set_xlabel('Days')

                    # if idx == len()
                    idx+=1

            if plt_key == 'pie':
                

                row = len(accorderies)
                
                col = len(titles)
                
                # print(row, col)
                # print(data)
            
                fig, axes = plt.subplots(row, col)
                
                axes = axes.flatten()

                
                idx = 0
                for iter in range(row):
                    title_i =0
                    for i, kd in enumerate(data):
                        if kd == 'metadata':
                            continue
                        # print(iter, idx, kd)
                        explode = np.zeros(len(data[kd][iter]))
                        explode[0] = 0.1
                        axes[idx].set_title(titles[title_i])
                        axes[idx].pie(data[kd][iter], explode=tuple(explode), autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.3, edgecolor='white'))
                        
                        centre_circle = plt.Circle((0, 0), 0.7, color='white')
                        
                        axes[idx].add_artist(centre_circle)

                        title_i+=1
                        idx+=1
            
            plt.tight_layout()
            
            pdf.savefig()
            plt.close()















# arg_parser = argparse.ArgumentParser()

# arg_parser.add_argument('-i', '--input', required=True)
# arg_parser.add_argument('-o', '--output', required=True)
# arg_parser.add_argument('-sh', '--sheet_name', default='Snapshot Average Metrics')

# args = arg_parser.parse_args()

# filters = args.__dict__

# plot_metrics_average(filters)
res = compute_bias_metrics(input_dir_path='data\\accorderies', net_ids=['109', '113'], snapshot_size = 365, super_star_threshold=.5)

bias_report(res)

# res = compute_bias_metrics()


# with open('output.txt', 'w') as filehandle:
#     json.dump(res, filehandle)
# print(res)