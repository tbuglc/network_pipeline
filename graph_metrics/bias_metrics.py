import argparse
import os
import sys
import json
from graph_loader import data_loader, load_accorderie_network
from snapshot_generator import create_snapshots
from utils import get_start_and_end_date,  add_sheet_to_xlsx, create_xlsx_file, save_csv_file, accorderies
from metrics import graph_novelty, super_stars_count, get_avg_in_out_degree, get_avg_in_out_disbalance, get_avg_weighted_in_out_degree, get_unique_edges_vs_total, node_attribute_variance
# accept input folder
from collections import OrderedDict
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
from math import ceil
from matplotlib.backends.backend_pdf import PdfPages
from dateutil import parser
import igraph as ig
from matplotlib.colors import Normalize, to_hex


def compute_bias_metrics(input_dir_path, net_ids=[], s_date='', e_date='', snapshot_size = 365, super_star_threshold=.5):
    '''
        net_ids: serves as a filter
    '''    
    result = OrderedDict()
    result["metadata"] = {}

    prob_scale = np.arange(0,1.1,0.1)
    result["metadata"]["snapshot_size"] = snapshot_size
    result["metadata"]["super_star_threshold"] = super_star_threshold
    result["metadata"]["accorderies"] = []
    result["metadata"]["dates"] = {} # key: accorderie, value is an array of start and end date
  
    result["Node Novelty"] = {"plt": "plot", "data": [], "scale": prob_scale}
    result["Edge Novelty"] = {"plt": "plot", "data": [], "scale": prob_scale}
    result["Weighted Node Novelty"] = {"plt": "plot", "data": [], "scale": prob_scale}
    result["In-Out Degree"] = {"plt": "bar", "data": [], "scale": prob_scale}
    result["In-Out Weigthed(hours) Degree"] = {"plt": "bar", "data": [], "scale": prob_scale}
    result["Disbalance"] = {"plt": "bar", "data": [], "scale": prob_scale[5:]}
    result["Unique Edges"] = {"plt": "bar", "data": [], "scale": []}

    result["Super Stars Sum"] = {"plt": "bar_label", "data": [], "scale": []}
    result["Super Stars In-deg"] = {"plt": "bar_label", "data": [], "scale": []}
    result["Super Stars Out-deg"] = {"plt": "bar_label", "data": [], "scale": []}
    result["Node Attribute Distances"] = {"plt": "graph", "data": [], "scale": []}

    start_date = None
    end_date = None

    # folders = None
    
    if len(net_ids) == 0:
        net_ids = os.listdir(input_dir_path)

    for fd in net_ids:
        print(f'\n====== {fd} ====== \n')
        count_super = 0

        if  '-' in fd:
            split_fd = fd.split('-')
            
            # incase -- are put infront
            split_fd.sort(reverse=True)

            count_super = len(split_fd[1:])
            fd = ''.join(split_fd[:1])



        # validate fd to avoid errors while reading graphs 
        if fd not in os.listdir(input_dir_path):
            print(f'skipping invalid folder id: {fd}')
            # skip ids not in folder of graphs
            # this may happen if the input id is not correct or by user error
            continue

        if s_date and not e_date:
            start_date =  parser.parse(s_date)
            _ , end_date = get_start_and_end_date(input_dir=os.path.join(input_dir_path, fd))
        elif e_date and not s_date:
            end_date =  parser.parse(e_date)
            start_date, _ = get_start_and_end_date(input_dir=os.path.join(input_dir_path, fd))
        elif not s_date and not e_date:
            start_date, end_date = get_start_and_end_date(input_dir=os.path.join(input_dir_path, fd))
        elif s_date and e_date:
            start_date =  parser.parse(s_date)
            end_date =  parser.parse(e_date)
        else:
            print('Nothing')
            continue 

        accorderie_name  = accorderies[int(fd)]
    
        print(f"Name: {accorderie_name}, date: {start_date} - {end_date}")

        if count_super > 0:
            result["metadata"]["accorderies"].append(f'{accorderie_name} #nodes - {count_super}')
        else:
            result["metadata"]["accorderies"].append(accorderie_name)

        result["metadata"]["dates"][accorderie_name] = [start_date, end_date]

        # try:    
        g = load_accorderie_network(os.path.join(input_dir_path, fd))

        if count_super > 0:
            for i in range(count_super):
                degrees = g.degree()
                node_with_highest_degree = degrees.index(max(degrees))
                g.delete_vertices(node_with_highest_degree)

        print('\nNode Novelty')
        _, _, node_novelty = graph_novelty(g, sn_size=snapshot_size, start_date=start_date, end_date=end_date)
        print('\nEdge Novelty')
        _, _, edge_novelty = graph_novelty(g, sn_size=snapshot_size, start_date=start_date, end_date=end_date, subset='EDGE')

        print('\nWeighted Node Novelty')
        _, _, weighted_node_novelty = graph_novelty(g, sn_size=snapshot_size, start_date=start_date, end_date=end_date, weighted=True)

        print('Super Start Sum')
        super_star_sum = super_stars_count(g, super_star_threshold, mode='all')
        print('Super Start In')
        super_star_in = super_stars_count(g, super_star_threshold, mode='in')
        print('Super Start Out')
        super_star_out = super_stars_count(g, super_star_threshold, mode='out')


        result['Node Novelty']["data"].append(node_novelty)
        result['Edge Novelty']["data"].append(edge_novelty)
        result['Weighted Node Novelty']["data"].append(weighted_node_novelty)

        print('Average In-Out')
        result["In-Out Degree"]["data"].append(get_avg_in_out_degree(g=g))
        print('In-Out Weigthed(hours) Degree')
        result["In-Out Weigthed(hours) Degree"]["data"].append(get_avg_in_out_disbalance(g=g) )
        print('Disbalance')
        result["Disbalance"]["data"].append(get_avg_weighted_in_out_degree(g=g) )
        print('Unique Edges')
        result["Unique Edges"]["data"].append(get_unique_edges_vs_total(g=g))
        
        print('ss sum')
        result['Super Stars Sum']["data"].append(super_star_sum)
        print('ss in')
        result['Super Stars In-deg']["data"].append(super_star_in)
        print('ss out')
        result['Super Stars Out-deg']["data"].append(super_star_out)
        print('ss node attribute')
        result["Node Attribute Distances"]["data"].append(node_attribute_variance(g))
        # except Exception as e:
        #     print(f'compute_bias_metrics error: {e}')
        #     break
    return result


def bias_report(metrics_data):
    if not metrics_data:
        return 
    
    accorderies = metrics_data['metadata']["accorderies"]
    report_name = '_'.join(accorderies).lower()
    
    with PdfPages(f'new_{report_name}_bias_report.pdf') as pdf:
        for key in metrics_data:

            if key == 'metadata':
                continue

            print(f'Key: {key}')
            plt_key = metrics_data[key]["plt"]

            data = metrics_data[key]["data"]
            # scale = metrics_data[key]["scale"]
            if len(data) == 0:
                continue

            # _key = key.lower()

            width = 0.5  # the width of the bars: can also be len(x) sequence
            
            plt.title(key)

            if plt_key == 'plot':
                tick_labels = []
                tick_X_label = []
                tick_positions = np.arange(0, 1.1, 0.1)
                for idx, d in enumerate(data):
                    labels = [k[0] for k in d ]  
                    X = [k[1] for k in d ]  
                    
                    X_label = np.arange(len(d)) 
                   
                    plt.plot(X_label, X, label=accorderies[idx])
                    
                    if len(labels) > len(tick_labels):
                        tick_labels = labels
                        tick_X_label = X_label

                plt.yticks(tick_positions)
                plt.xticks(tick_X_label, labels=tick_labels, rotation=45, ha='right')
                plt.legend()
            elif plt_key == 'bar_label':
                ax = None
                axes = []
                if len(data) > 1:
                    _, ax = plt.subplots(ceil(len(data) /2) ,2)
                    axes = ax.flatten()
                else:
                    _, ax = plt.subplots()
                    axes.append(ax)
            
                for idx, (total, count, result) in enumerate(data):
                    title = f'#nodes={count}, total={total}'
                    axes[idx].bar(np.arange(len(result)) + 1,result)
                    axes[idx].set_yticks(np.arange(0,1.1, 0.1))
                    if len(result) < 5:
                        axes[idx].set_xticks(np.arange(len(result)) + 1)

                    axes[idx].set_title(f'{accorderies[idx]}\n{key}')
                    axes[idx].set_xlabel(title)
                    axes[idx].set_ylabel('Proportion')
            
            elif plt_key == 'bar':
                for i,d in enumerate(data):
                    X = np.arange(len([d]))
                    X_new = X + (i - 1) * width
                    
                    plt.bar(X_new ,[d], width=width, label=accorderies[i])

                tick_positions = np.arange(0, 1.1, 0.1)

                plt.yticks(tick_positions)
                x_ticks_pos = np.arange(-width, len(data), width)
                plt.xticks(x_ticks_pos[:len(accorderies) ], accorderies)

                plt.ylabel('Proportion')
                plt.xlabel('Accorderies')
            elif plt_key == 'graph':
                ax = None
                axes = []
                print(data)
                for dt in data:
                    for d in dt:
                        print('graph: ',d.summary())
                        if len(d) > 1:
                            _, ax = plt.subplots(ceil(len(d) /2) ,2)
                            axes = ax.flatten()
                        else:
                            _, ax = plt.subplots()
                            axes.append(ax)

                        for idx, gd in enumerate(d):
                            
                            layout = gd.layout('fr')

                            edge_weights = gd.es["weight"]
                            
                            axes[idx].set_title(accorderies[idx])

                            norm = Normalize(vmin=min(edge_weights), vmax=max(edge_weights))

                            # # Create a colormap from the normalized edge weights
                            cmap = plt.cm.viridis

                            # # Map normalized edge weights to colors in the colormap
                            edge_colors = [to_hex(cmap(norm(weight))) for weight in edge_weights]


                            ig.plot(
                                gd,
                                target=axes[idx],
                                vertex_label=gd.vs['name'],
                                edge_width=np.array(gd.es['weight']) * 10,
                                edge_color=edge_colors, 
                                #  vertex_size=np.array(gd.degree()) / 10,
                                layout=layout
                            )

            plt.tight_layout()
            pdf.savefig()
            plt.close()

all_accorderies = {
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

# for ac in all_accorderies:
#     # print(ac)
#     for sh in ['109']:
res = compute_bias_metrics(input_dir_path='data\\accorderies', s_date='01/01/2015', net_ids=['109'], snapshot_size = 365, super_star_threshold=.5)

# bias_report(res)





# print(res['In-Out Weigthed(hours) Degree'])

# def compute_bias_metrics_per_page(input_dir_path, s_date='', e_date='', net_ids=[],  snapshot_size = 365, super_star_threshold=.5):
#     '''
#         net_ids: serves as a filter
#     '''    
#     result = OrderedDict()
#     result["first_page"] = {
#         "plt": 'plot',
#         "accorderies": [],
#         "snapshot_size": snapshot_size,
#         "titles": ["Node Novelty", "Edge Novelty", "Weighted Node Novelty"],
#         "data": {
#             "metadata": [],
#             "labels": [],
#             "Node Novelty": [],
#             "Edge Novelty": [],
#             "Weighted Node Novelty": []
#         }
#     }
#     result["second_page"] = {
#         "plt": 'bar',
#         "accorderies": [],
#         "snapshot_size": snapshot_size,
#         "titles": ["Global Metrics"],
#         "data": {
#             "metadata": [],
#             "labels": [],
#             "Global Metrics": []
#         }
#     }
#     result["third_page"] = {
#         "plt": 'pie',
#         "accorderies": [],
#         "snapshot_size": snapshot_size,
#         "titles": ["Super Stars Sum", "Super Stars In-deg", "Super Stars Out-deg"],
#         "data": {
#             "metadata": [],
#             "labels": [],
#             "Super Stars Sum": [],
#             "Super Stars In-deg": [],
#             "Super Stars Out-deg": [],
#         }
#     }


#     folders = []
#     if not input_dir_path and len(net_ids) > 0:
#         folders = net_ids
#     else:
#         folders = os.listdir(input_dir_path)
    
#     # print(folders, net_ids)
#     for fd in folders:
#         if len(net_ids) > 0 and fd not in net_ids:
#             # print('skipping' + fd)
#             continue

#         accorderie_name  = accorderies[int(fd)]
#         # print(net_ids)
#         print('starting: '+ accorderie_name)
#         result["first_page"]["accorderies"].append(accorderie_name)
#         result["second_page"]["accorderies"].append(accorderie_name)
#         result["third_page"]["accorderies"].append(accorderie_name)
#         # print("Name: " + accorderie_name)
#         # result[accorderie_name] = {}

#         try:
#             start_date = None
#             end_date = None

#             if s_date and not e_date:
#                 print('start date only', s_date)
#                 print('end', e_date)
#                 start_date =  parser.parse(s_date)
#                 _ , end_date = get_start_and_end_date(input_dir=os.path.join(input_dir_path, fd))
#             elif e_date and not s_date:
#                 print('end date only')

#                 end_date =  parser.parse(e_date)
#                 start_date, _ = get_start_and_end_date(input_dir=os.path.join(input_dir_path, fd))
#             elif not s_date and not e_date:
#                 print('not both date only')
#                 start_date, end_date = get_start_and_end_date(input_dir=os.path.join(input_dir_path, fd))
#             elif s_date and e_date:
#                 print('both', start_date, end_date)
#                 start_date =  parser.parse(s_date)
#                 end_date =  parser.parse(e_date)

#             else:
#                 print('Nothing')
#                 return 
            
#             g = load_accorderie_network(os.path.join(input_dir_path, fd))
            
#             result["first_page"]["data"]["metadata"].append({"start_date": start_date, "end_date": end_date })
#             result["second_page"]["data"]["metadata"].append({"start_date": start_date, "end_date": end_date })
#             result["third_page"]["data"]["metadata"].append({"start_date": start_date, "end_date": end_date })
            
#             _, sn_size, node_novelty = graph_novelty(g, sn_size=snapshot_size, start_date=start_date, end_date=end_date)
#             print('\nnode')
#             _, sn_size, edge_novelty = graph_novelty(g, sn_size=snapshot_size, start_date=start_date, end_date=end_date, subset='EDGE')
#             print('\nedge')

#             _, sn_size, weighted_node_novelty = graph_novelty(g, sn_size=snapshot_size, start_date=start_date, end_date=end_date, weighted=True)
#             print('\nweighted')

#             super_star_sum = super_stars_count(g=g, threshold=super_star_threshold, mode='all')
#             super_star_in = super_stars_count(g=g, threshold=super_star_threshold, mode='in')
#             super_star_out = super_stars_count(g=g, threshold=super_star_threshold, mode='out')
#             print('super stars')

            
#             global_metrics =[get_avg_in_out_degree(g=g) ,get_avg_in_out_disbalance(g=g) ,get_avg_weighted_in_out_degree(g=g) ,get_unique_edges_vs_total(g=g)]
            
#             result["first_page"]["data"]["labels"].append([nn[0] for nn in node_novelty])
#             result["first_page"]["data"]["Node Novelty"].append([nn[1] for nn in node_novelty])
#             result["first_page"]["data"]["Edge Novelty"].append([en[1] for en in edge_novelty])
#             result["first_page"]["data"]["Weighted Node Novelty"].append([wn[1] for wn in weighted_node_novelty])
            
#             result["second_page"]["data"]["Global Metrics"].append(global_metrics)
#             # print(accorderie_name, super_star_sum)
#             result["third_page"]["data"]["Super Stars Sum"].append(super_star_sum)
#             result["third_page"]["data"]["Super Stars In-deg"].append(super_star_in)
#             result["third_page"]["data"]["Super Stars Out-deg"].append(super_star_out)
            

#         except Exception as e:
#             print(e)
#             break
#         print('completing: '+ accorderie_name)
#     return result



# def bias_report_per_page(metrics_data):
#     # print(metrics_data)
#     if not metrics_data:
#         return 
    
#     accorderies = metrics_data['first_page']["accorderies"]

#     report_name = '_'.join(accorderies)
#     X_ticks = ["Average in-out degree", "In-out disbalance", "In-out-degree", "Unique edges"]
#     with PdfPages(f'{report_name}_bias_report.pdf') as pdf:
#         for key in metrics_data:
#             plt_key = metrics_data[key]["plt"]

#             data = metrics_data[key]["data"]
#             labels = metrics_data[key]["data"]["labels"]
#             titles = metrics_data[key]["titles"]
#             accorderies = metrics_data[key]["accorderies"]
            

#             idx = None
#             if plt_key == 'plot':
#                 row = ceil((len(data) - 1.01)/ 2)
#                 # print(data)
#                 col = ceil(len(metrics_data[key]["titles"])/2)
#                 # print(row, col)
#                 fig, axes = plt.subplots(row, col)

#                 max_label_idx = 0
#                 # label_idx = 0
#                 for i,label in enumerate(labels):
#                     if len(label) > len(label[max_label_idx]):
#                         max_label_idx = i
#                         # label_idx = i
                
                
#                 axes = axes.flatten()
#                 idx = 0
#                 for kd in data:

#                     if len(data[kd]) == 0:
#                         continue
#                     if kd == 'metadata' or kd == 'labels':
#                         continue
#                     axes[idx].set_title(titles[idx])
#                     for i,d in enumerate(data[kd]):
#                         X = np.arange(len(d))
                       
#                         axes[idx].plot(X,d)
#                     axes[idx].set_ylabel('Proportion')
#                     axes[idx].set_xlabel('Nodes''Days')
#                     axes[idx].set_xlabel('Nodes''Days')
#                     axes[idx].set_xticks(np.arange(len(labels[max_label_idx])), labels[max_label_idx],  rotation=45, ha='right')
#                     idx+=1


#             if plt_key == 'bar':
#                 row = ceil(len(data) - 1.01/ 2)
                
#                 col = len(metrics_data[key]["titles"])
            
#                 fig, axes = plt.subplots(row, col)

#                 axes = axes.flatten()
#                 idx = 0
#                 width = 0.5
#                 for kd in data:
#                     if len(data[kd]) == 0:
#                         continue
                    
#                     if kd == 'metadata':
#                         continue
             
#                     axes[idx].set_title(titles[idx])
#                     for i,d in enumerate(data[kd]):
#                         X = np.arange(len(d))
#                         X_new = X + (i - 1) * width
                        
#                         axes[idx].bar(X_new ,d, width=width)
#                         axes[idx].set_xticks(X_new, X_ticks)
                    
#                     axes[idx].set_ylabel('Proportion')
#                     axes[idx].set_xlabel('Nodes''Days')
#                     axes[idx].set_xlabel('Nodes''Days')

#                     idx+=1

#             if plt_key == 'pie':
#                 row = len(accorderies)
                
#                 col = len(titles)
            
#                 fig, axes = plt.subplots(row, col)
                
#                 axes = axes.flatten()

                
#                 idx = 0
#                 for iter in range(row):
#                     # print(iter)
#                     title_i =0
#                     for i, kd in enumerate(data):
#                         if len(data[kd]) == 0:
#                             continue

#                         if kd == 'metadata':
#                             continue
#                         # print(i)
#                         # if iter == 1:
#                         #     print(data[kd])
#                             # print()
#                         explode = np.zeros(len(data[kd][iter]))
#                         explode[0] = 0.1
#                         axes[idx].set_title(titles[title_i])
                        
#                         wedges, texts, _ = axes[idx].pie(data[kd][iter], explode=tuple(explode), autopct='%1.1f%%', startangle=90)

#                         # Check if there are multiple wedges to create the annotations
#                         # if len(data[kd][iter]) > 1:
#                         #     bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
#                         #     kw = dict(arrowprops=dict(arrowstyle="-"),
#                         #             bbox=bbox_props, zorder=0, va="center")

#                         #     for i, p in enumerate(wedges):
#                         #         ang = (p.theta2 - p.theta1)/2. + p.theta1
#                         #         y = np.sin(np.deg2rad(ang))
#                         #         x = np.cos(np.deg2rad(ang))
#                         #         horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
#                         #         connectionstyle = f"angle,angleA=0,angleB={ang}"
#                         #         kw["arrowprops"].update({"connectionstyle": connectionstyle})
#                         #         axes[idx].annotate(data[kd][iter][i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
#                         #                             horizontalalignment=horizontalalignment, **kw)

#                         title_i+=1
#                         idx+=1
#             plt.subplots_adjust(bottom=0.20)
#             plt.tight_layout()
            
#             pdf.savefig()
#             plt.close()


# idx = None
# if plt_key == 'plot':
#     row = ceil((len(data) - 1.01)/ 2)
#     # print(data)
#     col = ceil(len(metrics_data[key]["titles"])/2)
#     # print(row, col)
#     fig, axes = plt.subplots(row, col)

#     max_label_idx = 0
#     # label_idx = 0
#     for i,label in enumerate(labels):
#         if len(label) > len(label[max_label_idx]):
#             max_label_idx = i
#             # label_idx = i
    
    
#     axes = axes.flatten()
#     idx = 0
#     for kd in data:

#         if len(data[kd]) == 0:
#             continue
#         if kd == 'metadata' or kd == 'labels':
#             continue
#         axes[idx].set_title(titles[idx])
#         for i,d in enumerate(data[kd]):
#             X = np.arange(len(d))
            
#             axes[idx].plot(X,d)
#         axes[idx].set_ylabel('Proportion')
#         axes[idx].set_xlabel('Nodes''Days')
#         axes[idx].set_xlabel('Nodes''Days')
#         axes[idx].set_xticks(np.arange(len(labels[max_label_idx])), labels[max_label_idx],  rotation=45, ha='right')
#         idx+=1


# if plt_key == 'bar':
#     row = ceil(len(data) - 1.01/ 2)
    
#     col = len(metrics_data[key]["titles"])

#     fig, axes = plt.subplots(row, col)

#     axes = axes.flatten()
#     idx = 0
#     width = 0.5
#     for kd in data:
#         if len(data[kd]) == 0:
#             continue
        
#         if kd == 'metadata':
#             continue
    
#         axes[idx].set_title(titles[idx])
#         for i,d in enumerate(data[kd]):
#             X = np.arange(len(d))
#             X_new = X + (i - 1) * width
            
#             axes[idx].bar(X_new ,d, width=width)
#             axes[idx].set_xticks(X_new, X_ticks)
        
#         axes[idx].set_ylabel('Proportion')
#         axes[idx].set_xlabel('Nodes''Days')
#         axes[idx].set_xlabel('Nodes''Days')

#         idx+=1

# if plt_key == 'pie':
#     row = len(accorderies)
    
#     col = len(titles)

#     fig, axes = plt.subplots(row, col)
    
#     axes = axes.flatten()

    
#     idx = 0
#     for iter in range(row):
#         # print(iter)
#         title_i =0
#         for i, kd in enumerate(data):
#             if len(data[kd]) == 0:
#                 continue

#             if kd == 'metadata':
#                 continue
#             # print(i)
#             # if iter == 1:
#             #     print(data[kd])
#                 # print()
#             explode = np.zeros(len(data[kd][iter]))
#             explode[0] = 0.1
#             axes[idx].set_title(titles[title_i])
            
#             wedges, texts, _ = axes[idx].pie(data[kd][iter], explode=tuple(explode), autopct='%1.1f%%', startangle=90)

# Check if there are multiple wedges to create the annotations
# if len(data[kd][iter]) > 1:
#     bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
#     kw = dict(arrowprops=dict(arrowstyle="-"),
#             bbox=bbox_props, zorder=0, va="center")

#     for i, p in enumerate(wedges):
#         ang = (p.theta2 - p.theta1)/2. + p.theta1
#         y = np.sin(np.deg2rad(ang))
#         x = np.cos(np.deg2rad(ang))
#         horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
#         connectionstyle = f"angle,angleA=0,angleB={ang}"
#         kw["arrowprops"].update({"connectionstyle": connectionstyle})
#         axes[idx].annotate(data[kd][iter][i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
#                             horizontalalignment=horizontalalignment, **kw)

# title_i+=1
# idx+=1



















# arg_parser = argparse.ArgumentParser()

# arg_parser.add_argument('-i', '--input', required=True)
# arg_parser.add_argument('-o', '--output', required=True)
# arg_parser.add_argument('-sh', '--sheet_name', default='Snapshot Average Metrics')

# args = arg_parser.parse_args()

# filters = args.__dict__

# plot_metrics_average(filters)


# def compute_bias_metrics_og(input_dir_path, net_ids=[], snapshot_size = 365, super_star_threshold=.5):
#     '''
#         net_ids: serves as a filter
#     '''    

#     # result = {
#     #     "Snapshot Size": snapshot_size,
#     # }
#     result = OrderedDict()
#     result["metadata"] = {}

#     result["metadata"]["snapshot_size"] = snapshot_size,
#     result["metadata"]["super_star_threshold"] = super_star_threshold,
#     # result["metadata"]["start_date"] = start_date.strftime("%Y-%m-%d"),
#     # result["metadata"]["end_date"] = end_date.strftime("%Y-%m-%d"),
#     result["Node Novelty"] = {"plt": "plot", "data": []}
#     result["Edge Novelty"] = {"plt": "plot", "data": []}
#     result["Weighted Node Novelty"] = {"plt": "plot", "data": []}
#     result["Global Metrics"] = {"plt": "bar", "data": []}
#     result["Super Stars Sum"] = {"plt": "pie", "data": []}
#     result["Super Stars In-deg"] = {"plt": "pie", "data": []}
#     result["Super Stars Out-deg"] = {"plt": "pie", "data": []}


#     folders = []
#     if not input_dir_path and len(net_ids) > 0:
#         folders = net_ids
#     else:
#         folders = os.listdir(input_dir_path)
    
#     for fd in folders:

#         if len(net_ids) > 0 and fd not in net_ids:
#             continue

#         print('\n\n\n\nFOLDER: '+ fd)
#         accorderie_name  = accorderies[int(fd)]
    
#         print("Name: " + accorderie_name)
#         # result[accorderie_name] = {}

#         try:
#             start_date, end_date = get_start_and_end_date(input_dir=os.path.join(input_dir_path, fd))
#             g = load_accorderie_network(os.path.join(input_dir_path, fd))
            
#             # print(g.vs[0:10])

#             _, sn_size, node_novelty = graph_novelty(g, sn_size=snapshot_size, start_date=start_date, end_date=end_date)
#             print('\nnode')
#             _, sn_size, edge_novelty = graph_novelty(g, sn_size=snapshot_size, start_date=start_date, end_date=end_date, subset='EDGE')
#             print('\nedge')

#             _, sn_size, weighted_node_novelty = graph_novelty(g, sn_size=snapshot_size, start_date=start_date, end_date=end_date, weighted=True)
#             print('\nweighted')

#             super_star_sum = super_stars_count(g, super_star_threshold, mode='all')
#             super_star_in = super_stars_count(g, super_star_threshold, mode='in')
#             super_star_out = super_stars_count(g, super_star_threshold, mode='out')
#             print('super start')

#             # result[accorderie_name]['Node Novelty'] = node_novelty
#             # result[accorderie_name]['Edge Novelty'] = edge_novelty
#             # result[accorderie_name]['Weighted Node Novelty'] = weighted_node_novelty
#             # result[accorderie_name]['Super Stars'] = super_star

#             # result.append(collection)

#             result['Node Novelty']["data"].append(node_novelty)
#             result['Edge Novelty']["data"].append(edge_novelty)
#             result['Weighted Node Novelty']["data"].append(weighted_node_novelty)
#             global_metrics =[get_avg_in_out_degree(g=g) ,get_avg_in_out_disbalance(g=g) ,get_avg_weighted_in_out_degree(g=g) ,get_unique_edges_vs_total(g=g)]
#             result['Global Metrics']["data"].append(global_metrics)
#             result['Super Stars Sum']["data"].append(super_star_sum)
#             result['Super Stars In-deg']["data"].append(super_star_in)
#             result['Super Stars Out-deg']["data"].append(super_star_out)

#         except Exception as e:
#             print(e)
#             break
#     return result

#     #     # create snapshots
#     #     snapshots = create_snapshots(
#     #     g, start_date=start_date, end_date=end_date, span_days=int(span_days))


# def bias_report_og(metrics_data):

#     metadata = metrics_data['metadata']
#     x_value = ceil((len(metrics_data.keys()) + len(metrics_data['Super Stars']['data']) - 1)/2)

#     fig, ax = plt.subplots(x_value, 2,  figsize=(12, 8))

#     row = 0
#     col = 0
    
#     for key in metrics_data:
#         if key == 'metadata' or key == 'Global Metrics':
#             continue
        
#                 # row+=1
#         if col == 2:
#             col = 0
#             row+=1

#         data = metrics_data[key]['data']
#         plt_key = metrics_data[key]['plt']
#         Y = []
        
#         # mod = row % 2
#         target = ax[row,col]

#         if plt_key == 'pie':
#             for i,d in enumerate(data):
#                 target = ax[row + i , ( i + col )%2]

#                 target.pie(d)
               
#             # row+=1                  
#         else: 
#             for d in data:
#                 if plt_key == 'bar':
#                     Y = np.arange(0, 1, .1)
#                     target.bar(d, Y)
#                 else:
#                     target.plot(d)
#             target.set_title(key)
#             target.set_ylabel('Proportion')
#             target.set_xlabel('Nodes''Days')
#             target.set_xlabel('Nodes''Days')
        
#         col+=1

#     global_metrics = metrics_data['Global Metrics']['data']
#     # print(x_value)
#     # plt.subplot(2,1,2)
#     width = 0.5
#     for idx, gm in enumerate(global_metrics):
#         X = np.arange(len(gm)) + width
#         if idx == 0:
#             X = X - width
        
#         plt.bar(X + width, gm)
        
#     plt.savefig('./sherbrooke_bias_metrics.pdf', dpi=300)
#     plt.close()
