from math import ceil
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from metrics import compute_degree_distribution

def pdf_degree_distribution(complete_graph, snapshots, folder_name):

    with PdfPages('./output/'+folder_name+'.pdf') as pdf:
        # plot complete graph distribution
        fig, ax = plt.subplots()

        xag, yag = compute_degree_distribution(complete_graph)
        ax.bar(xag, yag)
        ax.set_title('Degree Distribution')
        ax.set_ylabel('Frequency')
        ax.set_xlabel('Degree')

        pdf.savefig()
        plt.close()

        col = 2

        snapshopt_len = ceil(len(snapshots)/2)

        if (snapshopt_len < 2):
            col = 1
        # print('col :'+str(col))
        for x in range(snapshopt_len):
            fig, (ax1, ax2) = plt.subplots(1, col)

            if (2*x < len(snapshots)):
                snp = snapshots[2*x]

                xa, ya = compute_degree_distribution(g=snp['subgraph'])
                ax1.set_title(snp['title'])
                ax1.set_ylabel('Frequency')
                ax1.set_xlabel('Degree')
                ax1.bar(xa, ya)

            if ((2*x) + 1) < len(snapshots):
                snp = snapshots[(2*x) + 1]

                xa, ya = compute_degree_distribution(g=snp['subgraph'])
                ax2.set_ylabel('Frequency')
                ax2.set_xlabel('Degree')
                ax2.set_title(snp['title'])
                ax2.bar(xa, ya)

            fig.dpi = 300

            # target.set_title(snp['title'])

            pdf.savefig()

            plt.close()
