import pandas as pd
import gurobipy as grb
import os
import numpy as np
from igraph import *


# Extracting the data
thepath = os.getcwd()
arcsData = pd.read_csv(thepath + "/arcs.csv", sep=';', header=None)
namesNodes = pd.read_csv(thepath + "/nodes.csv", sep=',', header=None)


# Preparing the data: setting up the graph and weight matrices
nbNodes = arcsData.iloc[:, 0].max()
arcsList = [(i, j) for i, j in zip(arcsData.iloc[:, 0], arcsData.iloc[:, 1])]
coordinates = [(i, j) for i, j in zip(namesNodes.iloc[:, 1], namesNodes.iloc[:, 2])]
weights = arcsData.iloc[:, 2]
originNode = 84
destinationNode = 116

m = grb.Model('subway')
arcs = m.addVars(arcsList, obj=weights, name='arcs')
m.addConstrs((arcs.sum('*', station) - arcs.sum(station, '*') == 0 for station in range(nbNodes) if station not in [originNode, destinationNode]))
m.addConstr(arcs.sum('*', originNode) - arcs.sum(originNode, '*') == 1)
m.addConstr(arcs.sum('*', destinationNode) - arcs.sum(destinationNode, '*') == -1)

m.optimize()

path = originNode
if m.status == grb.GRB.Status.OPTIMAL:
    print('***Optimal solution***')
    print('Minimum distance from', namesNodes.iloc[originNode - 1, 0], 'to',
          namesNodes.iloc[destinationNode - 1, 0], '\n', m.objVal)
    print(namesNodes.iloc[originNode - 1, 0], '(#%d)' % (originNode))
    solution = m.getAttr('x', arcs)
    while path != destinationNode:
        for arc in arcsList:
            if arc[1] == path and solution[arc] == 1:
                print(namesNodes.iloc[arc[0] - 1, 0], '(#%d)' % (arc[0]))
                path = arc[0]

g = Graph()
g.add_vertices(nbNodes + 1)
g.add_edges(arcsList)
g.delete_vertices(0)
layout = Layout(coordinates)
g.vs[originNode]["label"] = namesNodes.iloc[originNode - 1, 0]
g.vs[destinationNode]["label"] = namesNodes.iloc[destinationNode - 1, 0]
plot(g, layout=layout)

