## Gregory Dannay
## https://github.com/gdannay

import pandas as pd
from dijkstra import *
import os
import numpy as np


# Extracting the data
thepath = os.getcwd()
arcs = pd.read_csv(thepath + "/Paris/arcs.csv", sep=';', header=None).values
namesNodes = pd.read_csv(thepath + "/Paris/nodes.csv", sep=',', header=None)


# Preparing the data: setting up the graph and weight matrices
nbNodes = int(np.amax(arcs[:, 0]))
arcs[:, :-1] = arcs[:, :-1] - 1     # Correcting for the 0 index
nodes = arcs[:, :-1]
weights = arcs[:, 2]
graph = [nodes[nodes[:, 0] == i, 1] for i in range(nbNodes)]
w_matrix = np.zeros((len(weights), len(weights)))
for idx, i, j in zip(range(len(arcs)), arcs[:, 0], arcs[:, 1]):
    w_matrix[int(i), int(j)] = arcs[idx, 2]


# Call the algorithm
originMode = 84 - 1
destinationNode = 116 - 1
dist, prec = dijkstra(graph, w_matrix, originMode, destinationNode)


# Display the solution
print('Minimum distance from', namesNodes.iloc[originMode, 0], 'to',
      namesNodes.iloc[destinationNode, 0], '\n', dist[destinationNode])
path = destinationNode
pathList = [path]
print('\n\nPath:')
while path != originMode:
    path = prec[path]
    pathList.append(path)
step = 0
while pathList:
    path = pathList.pop()
    print(step, namesNodes.iloc[path, 0], '(#%d)' % (path + 1))
    step += 1
