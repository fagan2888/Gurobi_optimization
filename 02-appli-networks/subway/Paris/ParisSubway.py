import pandas as pd
import gurobipy as grb
import os
import numpy as np
import time
from igraph import *


# Extracting the data
thepath = os.getcwd()
arcsData = pd.read_csv(thepath + "/arcs.csv", sep=';', header=None)
namesNodes = pd.read_csv(thepath + "/nodes.csv", sep=',', header=None)


# Preparing the data: nodes and weight matrices
nbNodes = arcsData.iloc[:, 0].max()
arcsList = [(i, j) for i, j in zip(arcsData.iloc[:, 0], arcsData.iloc[:, 1])]
weights = arcsData.iloc[:, 2]
originNode = 84
destinationNode = 116

# Setting up the model
m = grb.Model('subway')
arcs = m.addVars(arcsList, obj=weights, name='arcs')
m.addConstrs((arcs.sum('*', station) - arcs.sum(station, '*') == 0 for station in range(nbNodes)
              if station not in [originNode, destinationNode]), name='constr')
m.addConstr(arcs.sum('*', originNode) - arcs.sum(originNode, '*') == 1, name='constr')
m.addConstr(arcs.sum('*', destinationNode) - arcs.sum(destinationNode, '*') == -1, name='constr')

m.optimize()

# Display the solution
path = originNode
pathList = []
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
                pathList.append(path)

# Setting up the graph and the visual style
g = Graph()
edges = [(i, j) for i, j in zip(arcsData.iloc[:, 0], arcsData.iloc[:, 1]) if i <= j]
coordinates = [(i, -j) for i, j in zip(namesNodes.iloc[:, 2], namesNodes.iloc[:, 3])]
g.add_vertices(nbNodes + 1)
g.add_edges(edges)
g.delete_vertices(0)
layout = Layout(coordinates)
visual_style = {}
visual_style["layout"] = layout
visual_style["vertex_color"] = ["SkyBlue2" for i in range(nbNodes) if i not in [originNode, destinationNode]]
visual_style["vertex_color"][originNode] = "firebrick2"
visual_style["vertex_color"][destinationNode] = "forestgreen"
visual_style["vertex_size"] = [10 for i in range(nbNodes) if i not in [originNode, destinationNode]]
visual_style["vertex_size"][originNode] = 40
visual_style["vertex_size"][destinationNode] = 40
visual_style["edge_arrow_size"] = 0
visual_style["margin"] = (-200, -300, -700, -1000)
visual_style["bbox"] = (1100, 700)
visual_style["vertex_label"] = [None for i in range(nbNodes) if i not in [originNode, destinationNode]]
visual_style["vertex_label"][originNode] = namesNodes.iloc[originNode - 1, 0]
visual_style["vertex_label"][destinationNode] = namesNodes.iloc[destinationNode - 1, 0]
plot(g, **visual_style)

# Show the optimal path with iGraph
path = originNode
if m.status == grb.GRB.Status.OPTIMAL:
    while path != destinationNode:
        path = pathList.pop(0)
        visual_style["vertex_size"][path] = 40
        visual_style["vertex_label"][path] = namesNodes.iloc[path - 1, 0]
        plot(g, **visual_style)
        visual_style["vertex_size"][path] = 10
        visual_style["vertex_label"][path] = None
        #time.sleep(0.5)

