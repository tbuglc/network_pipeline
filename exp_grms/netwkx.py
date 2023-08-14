import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import pandas as pd
import networkx as nx

# Load the statnet library
statnet = importr("statnet")

# Load edge list from CSV
edges_df = pd.read_csv("C:\\Users\\bugl2301\\projects\\school\\network_pipeline\\data\\accorderies\\109\\transactions.csv")
G = nx.from_pandas_edgelist(edges_df, 'source', 'target')

# Compute centrality measures using NetworkX
betweenness = nx.betweenness_centrality(G)
closeness = nx.closeness_centrality(G)
degree = dict(G.degree())

# Convert centrality measures to R vectors
betweenness_rvector = robjects.FloatVector(betweenness.values())
closeness_rvector = robjects.FloatVector(closeness.values())
degree_rvector = robjects.IntVector(degree.values())

# Create an R data frame with centrality measures
centrality_rdf = robjects.DataFrame({
    "betweenness": betweenness_rvector,
    "closeness": closeness_rvector,
    "degree": degree_rvector
})

# Convert NetworkX graph to an adjacency matrix
adj_matrix = nx.to_numpy_array(G)

# Convert the adjacency matrix to a Pandas DataFrame
adj_df = pd.DataFrame(adj_matrix, index=G.nodes, columns=G.nodes)

# Convert the Pandas DataFrame to an R data frame
adj_rdf = robjects.conversion.py2rpy(adj_df)

# Construct the formula
formula = "~ edges + nodematch('betweenness', diff=betweenness) + nodematch('closeness', diff=closeness) + nodematch('degree', diff=degree)"

# Fit ERGM model
fit = statnet.ergm(formula, response=adj_rdf, model=statnet.ergm_model())

# Print summary of the ERGM model
print(robjects.r["summary"](fit))
