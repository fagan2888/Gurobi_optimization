import pandas as pd
from dijkstra import *
import os
import numpy as np


# Extracting the data
thepath = os.getcwd()
arcs = pd.read_csv(thepath + "/NYC/arcs.csv", sep=',').values
namesNodes = pd.read_csv(thepath + "/NYC/nodes.csv", sep=',')


# Preparing the data: setting up the graph and weight matrices
nbNodes = int(np.amax(arcs[:, 0]))
arcs[:, :2] = arcs[:, :2] - 1     # Correcting for the 0 index

names = namesNodes.iloc[:, 0] + namesNodes.iloc[:, 6]
nodes = arcs[:, :2]
weights = arcs[:, 2]
graph = [nodes[nodes[:, 0] == i, 1] for i in range(nbNodes)]

w_matrix = np.zeros((len(weights), len(weights)))
for idx, i, j in zip(range(len(arcs)), arcs[:, 0], arcs[:, 1]):
    w_matrix[int(i), int(j)] = arcs[idx, 2]


# Call the algorithm
originMode = 452 - 1
destionationNode = 471 - 1
dist, prec = dijkstra(graph, w_matrix, originMode, destionationNode)


# Display the solution
print('Minimum distance from', names[originMode], 'to',
      names[destionationNode], '\n', dist[destionationNode])
path = destionationNode
step = 0
print('\n\nPath:')
while path != originMode:
    print(step, names[path], '(#%d)' % (path + 1))
    step += 1
    path = prec[path]
print(step, names[path], '(#%d)' % (path + 1))