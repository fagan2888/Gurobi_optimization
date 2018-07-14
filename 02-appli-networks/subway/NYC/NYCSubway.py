## Gregory Dannay
## https://github.com/gdannay

import pandas as pd
import os
import numpy as np
import gurobipy as grb
from collections import Counter
from igraph import *


# Extracting the data
thepath = os.getcwd()
arcsData = pd.read_csv(thepath + "/arcs.csv", sep=',').drop_duplicates(subset=['from_stop_nb', 'to_stop_nb']).values
namesNodes = pd.read_csv(thepath + "/nodes.csv", sep=',')


# Preparing the data: setting up the graph and weight matrices
nbNodes = int(np.amax(arcsData[:, 0]))
names = namesNodes.iloc[:, 0] + ' ' + namesNodes.iloc[:, 6]
arcsList = [(i, j) for i, j in zip(arcsData[:, 0], arcsData[:, 1])]
weights = arcsData[:, 3]
originNode = 452
destinationNode = 471

# Setting up the model
m = grb.Model('Subway')
arcs = m.addVars(arcsList, obj=weights, name='arcs')
m.addConstrs((arcs.sum('*', station) - arcs.sum(station, '*') == 0 for station in range(nbNodes)
              if station not in [originNode, destinationNode]), name='Constr')
m.addConstr(arcs.sum('*', originNode) - arcs.sum(originNode, '*') == 1, name='Constr')
m.addConstr(arcs.sum('*', destinationNode) - arcs.sum(destinationNode, '*') == -1, name='Constr')

m.optimize()

# Display the solution
path = originNode
pathList = []
step = 1
if m.status == grb.GRB.Status.OPTIMAL:
    print('***Optimal solution***')
    print('Minimum distance from', names[originNode - 1], 'to',
          names[destinationNode - 1], '\n', m.objVal)
    print('0 :', names[originNode - 1], '(#%d)' % originNode)
    solution = m.getAttr('x', arcs)
    while path != destinationNode:
        for arc in arcsList:
            if arc[1] == path and solution[arc] == 1:
                print(step, ':', names[arc[0] - 1], '(#%d)' % arc[0])
                path = arc[0]
                pathList.append(path)
                step += 1

# Setting up the graph and the visual style
g = Graph()
edges = [(i, j) for idx, (i, j) in enumerate(zip(arcsData[:, 0], arcsData[:, 1])) if (j, i) not in arcsList[:idx]]
coordinates = [(i, -j) for i, j in zip(namesNodes.iloc[:, 2], namesNodes.iloc[:, 3])]
g.add_vertices(nbNodes + 1)
g.add_edges(edges)
g.delete_vertices(0)
layout = Layout(coordinates)
visual_style = {}
visual_style["layout"] = layout
visual_style["vertex_color"] = ["SkyBlue2" for i in range(nbNodes)]
visual_style["vertex_color"][originNode - 1] = "firebrick2"
visual_style["vertex_color"][destinationNode - 1] = "forestgreen"
visual_style["vertex_size"] = [10 for i in range(nbNodes)]
visual_style["vertex_size"][originNode - 1] = 40
visual_style["vertex_size"][destinationNode - 1] = 40
visual_style["edge_arrow_size"] = 0
visual_style["margin"] = (-500, -500, -500, -500)
visual_style["bbox"] = (1100, 700)
visual_style["vertex_label"] = [None for i in range(nbNodes)]
visual_style["vertex_label"][originNode - 1] = namesNodes.iloc[originNode - 1, 0]
visual_style["vertex_label"][destinationNode - 1] = namesNodes.iloc[destinationNode - 1, 0]
plot(g, **visual_style)

# Show the optimal path with iGraph

path = originNode
if m.status == grb.GRB.Status.OPTIMAL:
    while path != destinationNode:
        path = pathList.pop(0)
        visual_style["vertex_size"][path - 1] = 40
        visual_style["vertex_label"][path- 1] = namesNodes.iloc[path - 1, 0]
        plot(g, **visual_style)
        visual_style["vertex_size"][path- 1] = 10
        visual_style["vertex_label"][path- 1] = None
        #time.sleep(0.5)
