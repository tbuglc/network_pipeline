import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import igraph as ig
from utils import load_accorderie_network
# Load the statnet library
statnet = importr("statnet")

g = load_accorderie_network("C:\\Users\\bugl2301\\projects\\school\\network_pipeline\\data\\accorderies\\109\\members.csv","C:\\Users\\bugl2301\\projects\\school\\network_pipeline\\data\\accorderies\\109\\transactions.csv") 
# Assuming you have an igraph object named 'network'
# Convert igraph object to an ergm.init object
# Load the ergm package in R

ergm = importr("ergm")

network_obj = robjects.conversion.py2rpy(g)


# Compute centrality measures using igraph
betweenness = g.betweenness()
closeness = g.closeness(mode="all")
katz = g.katz()
degree = g.degree(mode="all")

# Convert centrality measures to ergm.init objects
betweenness_init = robjects.r["ergm_init_vector"](betweenness)
closeness_init = robjects.r["ergm_init_vector"](closeness)
katz_init = robjects.r["ergm_init_vector"](katz)
degree_init = robjects.r["ergm_init_vector"](degree)

# Construct the formula
formula = robjects.Formula("~ edges + betweenness_init + closeness_init + katz_init + degree_init")
# fit = statnet.ergm(formula, init=ergm_obj)
# Fit ERGM model
fit = statnet.ergm(formula, init=network_obj)

# Print summary of the ERGM model
print(robjects.r["summary"](fit))
