from igraph import Graph
import igraph as ig
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from utils import add_sheet_to_xlsx, create_xlsx_file, save_csv_file


def plot_degree_distribution(xa, ya, folder_name='', title='', line=False, logscale=False):
    fig, ax = plt.subplots()

    if (line):
        ax.plot(xa, ya)
    else:
        ax.bar(xa, ya)

    ax.set_title(title)
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Degree')

    fig.savefig(folder_name + '.pdf', dpi=300)
    plt.close()


def plot_communities(g, layout, title):
    g1 = g.copy()

    components = g1.clusters()

    file_writer = create_xlsx_file('./output/'+title+'metrics')

    with PdfPages('./output/'+title+'graphs.pdf') as pdf:
        for cidx in range(len(components)):

            gc = components.subgraph(cidx)
            members = gc.get_vertex_dataframe()

            add_sheet_to_xlsx(file_writer=file_writer,
                              data=members, title='Cluster '+str(cidx), index=True)

            communities = gc.community_edge_betweenness()

            communities = communities.as_clustering()

            num_communities = len(communities)
            palette1 = ig.RainbowPalette(n=num_communities)
            for i, community in enumerate(communities):
                gc.vs[community]['color'] = i
                community_edges = gc.es.select(_within=community)
                community_edges['color'] = i

            target = plt.axes()
            target.set_title('Cluster '+str(cidx))

            ig.plot(
                communities,
                target=target,
                mark_groups=True,
                palette=palette1,
                vertex_label=gc.vs['age'],
                vertex_size=0.4,
                edge_width=0.6,
                layout=layout
            )

            pdf.savefig()
            plt.close()
        save_csv_file(file_writer=file_writer)


def plot_graph_snapshots(snapshots, layout, title, color_palette):
    with PdfPages('./output/'+title+'.pdf') as pdf:
        for x in range(len(snapshots)):

            snp = snapshots[x]
            sub_graph = snp['subgraph']

            plt.figure(dpi=300)

            target = plt.axes()
            target.set_title(snp['title'])

            ig.plot(
                sub_graph,
                target=target,
                vertex_size=0.4,
                vertex_label=sub_graph.vs['age'],
                edge_color=[color_palette['palette'].get(serv_color)
                            for serv_color in range(color_palette['length'])],
                edge_width=0.6,
                layout=layout,
                vertex_label_size=4,)

            pdf.savefig()
            plt.close()


def plot_complete_graph(g=Graph, layout=Graph.layout,  title='', color_palette={}):
    fig, ax = plt.subplots()

    ig.plot(
        g,
        target=ax,
        vertex_label=g.vs['age'],
        vertex_size=0.4,
        edge_width=0.6,
        layout=layout,
        edge_color=[color_palette['palette'].get(serv_color)
                    for serv_color in range(color_palette['length'])],
        vertex_label_size=4,
        edge_align_label=True)

    fig.savefig('./output/' + title + '.pdf', dpi=300)
    plt.close()


def plot_snapshot_metrics(data, title):
    if (data is None):
        return

    fig, ax = plt.subplots(5, 1)

    for i, d in enumerate(data):
        xx = data[d]
        target = ax[i]
        target.bar(range(len(data[d])), xx)

    fig.savefig('./output/' + title + '.pdf', dpi=300)
    plt.close()
