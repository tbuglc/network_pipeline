import igraph as ig 
from datetime import datetime


g = ig.Graph([(0,1), (0,2), (2,3)])

g.vs['date'] = [datetime(2020,5,19), datetime(2021,4,20), datetime(2022,1,12)]
g.vs['age'] = [1,2,3]

g = g.vs.select(age=2)

print(g)