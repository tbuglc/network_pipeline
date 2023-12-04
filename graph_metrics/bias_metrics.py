import argparse
from functools import reduce
import os
import sys
import json
from graph_loader import data_loader, load_accorderie_network
from snapshot_generator import create_snapshots
from utils import get_start_and_end_date,  add_sheet_to_xlsx, create_xlsx_file, save_csv_file, accorderies
from metrics import graph_novelty, growth_rate, disparity, super_stars_count, get_avg_in_out_degree, get_avg_in_out_disbalance, get_avg_weighted_in_out_degree, get_unique_edges_vs_total, node_attribute_variance, compute_blau_index
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
import textwrap
from cycler import cycler

# num_colors = 30
# colormap = plt.cm.magma  # Using the viridis colormap
# colors = [colormap(i) for i in np.linspace(0, 1, num_colors)]
colors = [
    "#E91E63",
    "#000000",
    "#1616C2",
    "#F4F48F",
    "#28cf81",
    "#00FFFF",
    "#FA3600",
    "#AA00FF",
    "#03612e",
    "#B0C4DE",
    "#8B0000",
    "#FFC400",
    "#EB00FF",
    "#7FFF00",
    "#6b4dbf",
    "#CD853F",
    "#00AAFF",
    "#20B2AA",
    "#ff0080",
    "#B0E0E6",
    "#e3ff2f",
    "#102c49",
    "#2F4F4F",
    "#5F9EA0",
    "#4682B4",
    "#FFDEAD",
    "#D2691E",
    "#CA1A30",
    "#00FFAA",
    "#F0E68C",
]

custom_cycler = cycler(color=colors)
plt.rc('axes', prop_cycle=custom_cycler)


def wrap_labels(labels, width):
    """Wrap labels at specified width."""
    wrapped_labels = []
    for label in labels:
        wrapped_labels.append('\n'.join(textwrap.wrap(label, width)))
    return wrapped_labels


def compute_bias_metrics(net_ids=[], alias=[], s_date='', e_date='', snapshot_size=365, super_star_threshold=.5):
    '''
        net_ids: serves as a filter
    '''
    result = OrderedDict()
    result["metadata"] = {}

    prob_scale = np.arange(0, 1.1, 0.1)
    result["metadata"]["snapshot_size"] = snapshot_size
    result["metadata"]["super_star_threshold"] = super_star_threshold
    result["metadata"]["accorderies"] = []
    # key: accorderie, value is an array of start and end date
    result["metadata"]["dates"] = {}

    # result["Node Scatter"] = {"plt": "scatter",
    #                           "data": [], "scale": prob_scale}
    # result["Edge Scatter"] = {"plt": "scatter",
    #                           "data": [], "scale": prob_scale}
    # result["Weighted Scatter"] = {
    #     "plt": "scatter", "data": [], "scale": prob_scale}

    result["Node Novelty"] = {"plt": "plot",
                              "data": [], "scale": prob_scale, "label": [], 'ylabel': '', 'xlabel': ''}

    result["Weighted Node Novelty"] = {
        "plt": "plot", "data": [], "scale": prob_scale, "label": [], 'ylabel': '', 'xlabel': ''}

    result["Edge Novelty"] = {"plt": "plot",
                              "data": [], "scale": prob_scale, "label": [], 'ylabel': '', 'xlabel': ''}
    result["Number of Interactions"] = {
        "plt": "plot", "data": [], "scale": prob_scale, "label": [], 'ylabel': '', 'xlabel': ''}

    result["In-Out Degree"] = {"plt": "bar",
                               "data": [], "scale": prob_scale, "label": [], 'ylabel': '', 'xlabel': ''}
    result["In-Out Degree Box"] = {"plt": "box",
                                   "data": [], "scale": prob_scale, "label": [], 'ylabel': '', 'xlabel': ''}
    result["In-Out Degree Line"] = {"plt": "line",
                                    "data": [], "scale": prob_scale, "label": [], 'ylabel': '', 'xlabel': ''}

    result["In-Out Weigthed(hours) Degree"] = {"plt": "bar",
                                               "data": [], "scale": prob_scale, "label": [], 'ylabel': '', 'xlabel': ''}
    result["Member Growth Rate"] = {"plt": "line",
                                    "data": [], "scale": prob_scale, "label": [], 'ylabel': '', 'xlabel': ''}
    result["Interaction Growth Rate"] = {"plt": "line",
                                         "data": [], "scale": prob_scale, "label": [], 'ylabel': '', 'xlabel': ''}

    result["Disparity in-degree"] = {"plt": "bar",
                                     "data": [], "scale": prob_scale, "label": [], 'ylabel': '', 'xlabel': ''}
    result["Disparity out-degree"] = {"plt": "bar",
                                      "data": [], "scale": prob_scale, "label": [], 'ylabel': '', 'xlabel': ''}
    result["Disparity in+out-degree"] = {"plt": "bar",
                                         "data": [], "scale": prob_scale, "label": [], 'ylabel': '', 'xlabel': ''}
    # Heterogeneity
    result["Age Heterogeneity Index"] = {"plt": "bar",
                                         "data": [], "scale": prob_scale, "label": [], 'ylabel': '', 'xlabel': ''}
    result["Region Heterogeneity Index"] = {"plt": "bar",
                                            "data": [], "scale": prob_scale, "label": [], 'ylabel': '', 'xlabel': ''}
    result["Revenu Heterogeneity Index"] = {"plt": "bar",
                                            "data": [], "scale": prob_scale, "label": [], 'ylabel': '', 'xlabel': ''}
    result["Gender Heterogeneity Index"] = {"plt": "bar",
                                            "data": [], "scale": prob_scale, "label": [], 'ylabel': '', 'xlabel': ''}

    result["Disbalance"] = {"plt": "bar", "data": [],
                            "scale": prob_scale[5:], "label": [], 'ylabel': '', 'xlabel': ''}

    result["Unique Edges"] = {"plt": "bar",
                              "data": [], "scale": [], "label": [], 'ylabel': '', 'xlabel': ''}
    result["Density"] = {"plt": "line",
                         "data": [], "scale": [], "label": [], 'ylabel': '', 'xlabel': ''}
    result["Super Stars Sum"] = {"plt": "pie",
                                 "data": [], "scale": [], "label": [], 'ylabel': '', 'xlabel': ''}
    result["Super Stars In-deg"] = {"plt": "pie",
                                    "data": [], "scale": [], "label": [], 'ylabel': '', 'xlabel': ''}
    result["Super Stars Out-deg"] = {"plt": "pie",
                                     "data": [], "scale": [], "label": [], 'ylabel': '', 'xlabel': ''}

    result["Node Attribute Distances"] = {
        "plt": "heatmap", "data": [], "scale": [], "label": [], 'ylabel': '', 'xlabel': ''}

    start_date = None
    end_date = None

    if len(net_ids) == 0:
        # net_ids = os.listdir(input_dir_path)
        # print('NET IDS', net_ids)
        raise 'No network to compute metrics for'
    for fidx, fd in enumerate(net_ids):
        print(f'\n====== {fd} ====== \n')
        count_super = 0

        if '-' in fd:
            split_fd = fd.split('-')

            # incase -- are put infront
            split_fd.sort(reverse=True)

            count_super = len(split_fd[1:])
            fd = ''.join(split_fd[:1])

        if s_date and not e_date:
            start_date = parser.parse(s_date)
            _, end_date = get_start_and_end_date(
                input_dir=os.path.join(fd))
        elif e_date and not s_date:
            end_date = parser.parse(e_date)
            start_date, _ = get_start_and_end_date(
                input_dir=os.path.join(fd))
        elif not s_date and not e_date:
            start_date, end_date = get_start_and_end_date(
                input_dir=os.path.join(fd))
        elif s_date and e_date:
            start_date = parser.parse(s_date)
            end_date = parser.parse(e_date)
        else:
            print('Nothing')
            continue

        accorderie_name = alias[fidx]

        print(f"Name: {accorderie_name}, date: {start_date} - {end_date}")

        if count_super > 0:
            accorderie_name = f'{accorderie_name}'
            result["metadata"]["accorderies"].append(accorderie_name)
        else:
            result["metadata"]["accorderies"].append(accorderie_name)

        result["metadata"]["dates"][accorderie_name] = [start_date, end_date]
        print('START DATE: ', start_date, 'END DATE: ', end_date)
        # try:
        g = load_accorderie_network(os.path.join(fd))

        # Skip empty graphs
        if len(g.vs) == 0 or len(g.es) == 0:
            continue

        if count_super > 0:
            for _ in range(count_super):
                degrees = g.degree()
                node_with_highest_degree = degrees.index(max(degrees))
                g.delete_vertices(node_with_highest_degree)

        # get accorderie id if exist
        if '\\' in fd:
            acc_id = fd.split('\\')[-1:]
        elif '/' in fd:
            acc_id = fd.split('/')[-1:]
        else:
            acc_id = ""

        print('\nNode Grow Rate')
        gwth_rate, gwth_raw_rate = growth_rate(g, sn_size=snapshot_size, start_date=start_date,
                                               end_date=end_date, id=acc_id)
        result['Member Growth Rate']['data'].append([v for _, v in gwth_rate])
        result['Member Growth Rate']["xlabel"] = 'Date(s)'
        result['Member Growth Rate']["label"].append([d for d, _ in gwth_rate])
        # print(gwth_rate)
        # print(gwth_raw_rate)
        print('\Edge Grow Rate')
        e_gwth_rate, e_gwth_raw_rate = growth_rate(g, sn_size=snapshot_size, start_date=start_date,
                                                   end_date=end_date, id=acc_id, subset='EDGE')

        result['Interaction Growth Rate']['data'].append(
            [v for _, v in e_gwth_rate])
        result['Interaction Growth Rate']["xlabel"] = 'Date(s)'
        result['Interaction Growth Rate']["label"].append(
            [d for d, _ in e_gwth_rate])
        # print(e_gwth_rate)
        # print(e_gwth_raw_rate)

        print('\nDisparity in-degree')
        in_avg_disp, in_disp = disparity(g, mode='in')
        result['Disparity in-degree']['data'].append(in_avg_disp)
        # print(in_avg_disp)
        # print(in_disp)
        print('\nDisparity out-degree')
        out_avg_disp, out_disp = disparity(g, mode='out')
        result['Disparity out-degree']['data'].append(out_avg_disp)
        # print(out_avg_disp)
        # print(out_disp)
        print('\nDisparity ALL')
        all_avg_disp, all_disp = disparity(g, mode='all')
        result['Disparity in+out-degree']['data'].append(all_avg_disp)
        # print(all_avg_disp)
        # print(all_disp)

        region_idx_blau = compute_blau_index(g, 'region')
        result['Region Heterogeneity Index']['data'].append(region_idx_blau)

        genre_idx_blau = compute_blau_index(g, 'genre')
        result['Gender Heterogeneity Index']['data'].append(genre_idx_blau)

        revenu_idx_blau = compute_blau_index(g, 'revenu')
        result['Revenu Heterogeneity Index']['data'].append(revenu_idx_blau)

        age_idx_blau = compute_blau_index(g, 'age')
        result['Age Heterogeneity Index']['data'].append(age_idx_blau)

        print('\nNode Novelty')
        _, _, node_novelty, raw_node_result, densities = graph_novelty(
            g, sn_size=snapshot_size, start_date=start_date, end_date=end_date, id=acc_id, density=True)
        print('ratio: ', node_novelty)
        print('raw: ', raw_node_result)
        print('\nEdge Novelty')
        _, _, edge_novelty, raw_edge_result, _ = graph_novelty(
            g, sn_size=snapshot_size, start_date=start_date, end_date=end_date, id=acc_id, subset='EDGE')
        print('ratio: ', edge_novelty)
        print('raw: ', raw_edge_result)
        print('\nWeighted Node Novelty')
        _, _, weighted_node_novelty, raw_weighted_result, _ = graph_novelty(
            g, sn_size=snapshot_size, start_date=start_date, end_date=end_date, id=acc_id, weighted=True)
        print('ratio: ', weighted_node_novelty)
        print('raw: ', raw_weighted_result)
        print('Super Start Sum')
        super_star_sum = super_stars_count(g, super_star_threshold, mode='all')
        print('Super Start In')
        super_star_in = super_stars_count(g, super_star_threshold, mode='in')
        print('Super Start Out')
        super_star_out = super_stars_count(g, super_star_threshold, mode='out')

        # result['Node Scatter']["data"].append(raw_node_result)
        # result['Edge Scatter']["data"].append(raw_edge_result)
        # result['Weighted Scatter']["data"].append(raw_weighted_result)

        result['Node Novelty']["data"].append(node_novelty)
        result['Edge Novelty']["data"].append(edge_novelty)
        edg_count = [(d, e) for _, e, d in raw_edge_result]
        # num_edg_count = [e for _, e, d in raw_edge_result]
        # nec_max = np.max(num_edg_count)
        # nec_min = np.min(num_edg_count)
        # num_edg_count = (np.array(num_edg_count) -
        #                  nec_min) / (nec_max - nec_min)
        # edg_count = [(label_edg_count[i], num_edg_count[i])
        #              for i in range(len(num_edg_count))]
        # print("EDGE COUNT", edg_count)
        # print('EDGE COUNT NORMALIZED', edg_count)

        result['Number of Interactions']["data"].append(edg_count
                                                        )
        print('density', densities)
        result['Density']["data"].append(densities
                                         )
        result['Density']["xlabel"] = 'Date(s)'
        result['Density']["label"].append([d for d, _ in node_novelty])
        result['Weighted Node Novelty']["data"].append(weighted_node_novelty)

        print("Average In-Out")
        res, array_res = get_avg_in_out_degree(g=g)
        # print(array_res)
        result["In-Out Degree"]["data"].append(res)
        result["In-Out Degree Box"]["data"].append(array_res)
        result["In-Out Degree Line"]["data"].append(array_res)

        print("Disbalance")
        result["Disbalance"]["data"].append(
            get_avg_in_out_disbalance(g=g))

        print("In-Out Weigthed(hours) Degree")
        result["In-Out Weigthed(hours) Degree"]["data"].append(
            get_avg_weighted_in_out_degree(g=g))

        print('Unique Edges')
        result["Unique Edges"]["data"].append(get_unique_edges_vs_total(g=g))

        print('Super Stars Sum')
        result['Super Stars Sum']["data"].append(super_star_sum)
        print('Super Stars In-deg')
        result['Super Stars In-deg']["data"].append(super_star_in)
        print('Super Stars Out-deg')
        result['Super Stars Out-deg']["data"].append(super_star_out)
        print('Node Attribute Distances')

        if 'random' not in accorderie_name.lower():
            result["Node Attribute Distances"]["data"].append(
                node_attribute_variance(g, accorderie_name))

        # except Exception as e:
        #     print(f'compute_bias_metrics error: {e}')
        #     break

    # rescale novelties to have the same date
    # 1. find the lowest date
    # 2. fill in missing values
    return result


def custom_autopct(pct):
    if pct > 2.5:
        return f'{pct:.1f}%'
    else:
        return ''


def bias_report(metrics_data):
    if not metrics_data:
        return

    accorderies = metrics_data['metadata']["accorderies"]
    report_name = '_'.join(accorderies).lower()

    with PdfPages(f'3_accorderies_report_with_stars.pdf') as pdf:
        for key in metrics_data:

            if key == 'metadata':
                continue

            plt_key = metrics_data[key]["plt"]

            data = metrics_data[key]["data"]

            if len(data) == 0:
                continue

            width = 0.5

            plt.title(key)

            if plt_key == 'plot':
                tick_labels = []
                tick_X_label = []
                tick_positions = np.arange(0, 1.1, 0.1)
                for idx, d in enumerate(data):
                    labels = [k[0] for k in d]
                    X = [k[1] for k in d]

                    X_label = np.arange(len(d))

                    plt.plot(X_label, X, label=wrap_labels(
                        [accorderies[idx]], 15)[0])

                    if len(labels) > len(tick_labels):
                        tick_labels = labels
                        tick_X_label = X_label
                # X_bool = [True if z <= 1 else False for z in X]

                if reduce(lambda v, w: v & w, [True if z <= 1 else False for z in X]):
                    plt.yticks(tick_positions)

                plt.xticks(tick_X_label, labels=tick_labels,
                           rotation=45, ha='right')

                plt.ylabel('Proportion')
                plt.xlabel('Date(s)')
                plt.subplots_adjust(right=0.7)
                plt.legend(loc='upper left', bbox_to_anchor=(
                    1, 1), fontsize='small')
                # plt.legend()
                plt.tight_layout()
                pdf.savefig()
                plt.close()
            elif plt_key == 'line':
                label = metrics_data[key]["label"]
                xlabel = metrics_data[key]["xlabel"]
                for idx, d in enumerate(data):
                    d.sort(reverse=True)

                    plt.plot(d, label=wrap_labels(
                        [accorderies[idx]], 15)[0])

                if len(label) > 0:
                    print(label[0], d)
                    plt.xticks(np.arange(len(d)), labels=label[0],
                               rotation=45, ha='right')

                plt.ylabel('Proportion')
                plt.xlabel(xlabel if xlabel else '# of members')
                plt.subplots_adjust(right=0.7)
                plt.legend(loc='upper left', bbox_to_anchor=(
                    1, 1), fontsize='small')
                plt.tight_layout()
                pdf.savefig()
                plt.close()
                # labels = [k[0] for k in d]
                # X = [k[1] for k in d]
                # print('d ===>', d)
                # X_label = np.arange(len(d))
                # print('d ===>', d)
                # plt.plot(X_label, X, label=accorderies[idx])
                # print()

            elif plt_key == 'box':
                plt.boxplot(data, notch=False, showmeans=True, meanline=True,
                            vert=True, showbox=True)

                plt.xticks([i + 1 for i in range(len(accorderies))],
                           [wrap_labels([ac], 15)[0] for ac in accorderies],  rotation=45, ha='right')
                plt.ylabel('Proportion')
                plt.xlabel('Accorderie(s)')
                plt.tight_layout()
                pdf.savefig()
                plt.close()

                # plt.close()
                # tick_labels = []
                # tick_X_label = []
                # tick_positions = np.arange(0, 1.1, 0.1)
                # fig, axes = plt.subplots()
                # axes = axes.flatten()
                # for idx, d in enumerate(data):
                # labels = [k[0] for k in d]
                # X = [k[1] for k in d]
                # print('d ===>', d)
                # X_label = np.arange(len(d))
                # print('d ===>', d)
                # plt.plot(X_label, X, label=accorderies[idx])
                # if len(labels) > len(tick_labels):
                #     tick_labels = labels
                #     tick_X_label = X_label
                # X_bool = [True if z <= 1 else False for z in X]

                # if reduce(lambda v, w: v & w, [True if z <= 1 else False for z in X]):
                #     plt.yticks(tick_positions)

                # plt.xticks(tick_X_label, labels=tick_labels,
                #    rotation=45, ha='right')
                # plt.legend()

            elif plt_key == 'heatmap':
                for dt_indx, dta in enumerate(data):
                    for dt in dta:
                        labls = dt["labels"]
                        title = dt["title"]
                        d = dt["data"]

                        for k in d:
                            # Create a heatmap from the data, using the 'Reds' colormap
                            plt.imshow(d[k], cmap='Reds', aspect='auto')
                            # Create colorbar
                            plt.colorbar()

                            # Show the data values on the heatmap
                            max_value = np.array(d[k]).max()
                            for i in range(d[k].shape[0]):
                                for j in range(d[k].shape[1]):
                                    # Choosing text color based on data value to ensure visibility. You might need to choose colors based on your data range and colormap.
                                    text_color = "black" if (d[k][i, j]) < (
                                        max_value / 3) else "white"
                                    v = f"{d[k][i, j]:.1f}"

                                    plt.text(j, i, v,
                                             ha="center", va="center", color=text_color)

                            # 'viridis' is a colormap; choose one you prefer

                            # Add a colorbar to the plot
                            # cbar = plt.colorbar()
                            # # Label for the colorbar
                            # cbar.set_label('Color Scale')
                            # Annotations for the legend
                            # plt.annotate('Low Value', (0, -0.5), color='black', fontsize=10)
                            # plt.annotate('High Value', (9, -0.5), color='black', fontsize=10)
                            # Label the x-axis values

                            plt.xticks(range(d[k].shape[1]),
                                       labls, rotation=45)
                            # Label the y-axis values
                            plt.yticks(range(d[k].shape[0]),
                                       labls, rotation=45)
                            # Set the title
                            plt.title(f'{accorderies[dt_indx]} <> {title}-{k}')
                            # plt.legend()
                            plt.tight_layout()
                            pdf.savefig()
                            plt.close()
            elif plt_key == 'scatter':
                for idx, d in enumerate(data):
                    X = []
                    Y = []
                    L = []
                    for x, y, l in d:
                        X.append(x)
                        Y.append(y)
                        L.append(l)

                    # Add labels for specific points
                    for i in range(len(X)):
                        plt.text(X[i], Y[i], L[i], fontsize=12,
                                 ha='center', va='bottom', rotation=90)

                    plt.scatter(X, Y, label=accorderies[idx])

                    # if len(labels) > len(tick_labels):
                    #     tick_labels = labels
                    #     tick_X_label = X_label

                # plt.yticks(tick_positions)
                # plt.xticks(tick_X_label, labels=tick_labels,
                #            rotation=45, ha='right')
                # plt.legend()
                plt.subplots_adjust(right=0.7)
                plt.legend(loc='upper left', bbox_to_anchor=(
                    1, 1), fontsize='small')

                plt.tight_layout()
                pdf.savefig()
                plt.close()
            elif plt_key == 'pie':
                axes = []

                for idx, (total, count, result, threshold) in enumerate(data):
                    fig, axes = plt.subplots()

                    title = f'#nodes={count}, total={total}, threshold={threshold}'
                    explode = np.zeros(len(result))
                    explode[0] = 0.1
                    # explode[len(result) - 1] = 0.1
                    labels = ["{:.2f}%".format(
                        rt * 100) if rt*100 > 2.5 else '' for rt in result]

                    # print('\n')
                    # print(result, explode)
                    if np.sum(result) == 0:
                        explode = np.zeros(len(result))
                    # print('\n')
                    # print(result, explode)

                    axes.pie(result, explode=explode,
                             autopct=custom_autopct,
                             shadow=True, startangle=135)

                    axes.set_title(f'{accorderies[idx]}\n{key}')

                    axes.set_xlabel(title)

                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()
            elif plt_key == 'bar':
                for i, d in enumerate(data):
                    X = np.arange(len([d]))
                    X_new = X + (i - 1) * width
                    # print('KEY: ', key, 'value: ', d)
                    plt.bar(X_new, [d], width=width, label=accorderies[i])

                tick_positions = np.arange(0, 1.1, 0.1)

                if 'disbalance' in key.lower():
                    tick_positions = np.arange(0.5, 1.1, .1)

                    plt.ylim(0.5, 1.1)

                plt.yticks(tick_positions)
                x_ticks_pos = np.arange(-width, len(data), width)
                #  plt.xticks(tick_X_label, labels=tick_labels,
                #            rotation=45, ha='right')
                plt.xticks(x_ticks_pos[:len(accorderies)],
                           [wrap_labels([acx], 15)[0] for acx in accorderies],  rotation=45, ha='right')

                plt.ylabel('Proportion')
                plt.xlabel('Accorderies')

                plt.tight_layout()
                pdf.savefig()
                plt.close()
            elif plt_key == 'graph':
                for idt, dt in enumerate(data):
                    axes = []

                    for idx, ob in enumerate(dt):
                        fig, axes = plt.subplots()
                        t, gd = ob.popitem()
                        layout = gd.layout('kk')

                        try:
                            if len(gd.es['weight']) > 0:
                                edge_weights = gd.es["weight"]
                        except Exception as e:
                            pass

                        axes.set_title(f'{accorderies[idt]} {t.title()}')

                        edge_colors = []
                        if len(edge_weights) > 0:
                            norm = Normalize(
                                vmin=min(edge_weights), vmax=max(edge_weights))

                            # # Create a colormap from the normalized edge weights
                            cmap = plt.cm.viridis

                            # # Map normalized edge weights to colors in the colormap
                            edge_colors = [to_hex(cmap(norm(weight)))
                                           for weight in edge_weights]
                        # print(f'{accorderies[idt]} {t.title()}')
                        # print(gd.vs['name'])
                        ig.plot(
                            gd,
                            target=axes,
                            vertex_label=gd.vs['name'],
                            # edge_width=edge_weights,
                            # edge_color=edge_colors,
                            # vertex_size=1,
                            # vertex_label_size=.8,
                            layout=layout
                        )

                        # plt.tight_layout()
                        pdf.savefig()
                        plt.close()

                plt.close()


all_accorderies = {
    2: "Québec",
    86: "Trois-Rivières",
    88: "Mercier-Hochelaga-M.",
    92: "Shawinigan",
    104: "Montréal-Nord",
    108: "Rimouski-Neigette",
    109: "Sherbrooke",
    110: "La Matanie",
    111: "Portneuf",
    112: "Granit",
    113: "Montréal-Nord secteur Nord-Est",
    114: "Rosemont",
    115: "Longueuil",
    116: "Réseau Accorderie (du Qc)",
    117: "La Manicouagan",
    118: "La Matapédia",
    119: "Grand Gaspé",
    120: "Granby et région",
    121: "Yukon",
}

#  s_date='01/01/2014', e_date='01/01/2022'
res = compute_bias_metrics(
    net_ids=[
        'accorderies\\2',
        'accorderies\\2-',

        'accorderies\\92',
        'accorderies\\92-',

        'accorderies\\109',
        'accorderies\\109-',

        # 'data\\accorderies\\86',
        # 'data\\accorderies\\88',
        # 'data\\accorderies\\104',
        # 'data\\accorderies\\108',
        # 'data\\accorderies\\110',
        # 'data\\accorderies\\111',
        # 'data\\accorderies\\112',
        # 'data\\accorderies\\113',
        # 'data\\accorderies\\114',
        # 'data\\accorderies\\115',
        # 'data\\accorderies\\117',
        # 'data\\accorderies\\118',
        # 'data\\accorderies\\119',
        # 'data\\accorderies\\120',
        # 'data\\accorderies\\121',
    ], s_date='28/02/2014',
    e_date='30/04/2022',
    alias=[
        "Québec",
        "Québec-1",

        "Shawinigan",
        "Shawinigan-1",

        "Sherbrooke",
        "Sherbrooke-1",

        # "Trois-Rivières",
        # "M-Hochelaga",
        # "Mntrl-Nord",
        # "Rimouski-Ngtte",
        # "La Matanie",
        # "Portneuf",
        # "Granit",
        # "Mntrl-S_N-E",
        # "Rosemont",
        # "Longueuil",
        # "La Manicouagan",
        # "La Matapédia",
        # "Grand Gaspé",
        # "Granby",
        # "Yukon",
    ], snapshot_size=365, super_star_threshold=.5)

# print('RESULT', res["Node Attribute Distances"]["data"])
bias_report(res)

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
