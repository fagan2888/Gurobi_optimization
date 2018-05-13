import pandas as pd
import numpy as np
import gurobipy as grb
import os

# Setting up the data
thepath = os.getcwd()
filename = "/distances.csv"
data = pd.read_csv(thepath + filename, sep=',')
dists = data.iloc[0:68, 1:11].fillna(-1).values
p = data.iloc[0:68, 11].values
q = data.iloc[68, 1:11].values
costs = dists.T[np.where(dists.T >= 0)]
name_source = list(data)
name_source = name_source[1:-1]
name_dest = data.iloc[:-1, 0]
roads = [(i, j) for col, i in enumerate(name_source) for row, j in enumerate(name_dest) if dists[row, col] >= 0]

# Setting up the model
m = grb.Model("Soviet")
arcs = m.addVars(roads, obj=costs, name="arcs")
m.addConstrs((arcs.sum(source, '*')) == q[idx] for idx, source in enumerate(name_source))
m.addConstrs((arcs.sum('*', dest) == p[idx] for idx, dest in enumerate(name_dest)))

# Display optimal solution
m.optimize()

if m.status == grb.GRB.Status.OPTIMAL:
    solution = m.getAttr('x', arcs)
    print('***Optimal solution***')
    for road in roads:
        if solution[road] > 0:
            print("%s -> %s : %g" % (road[0], road[1], solution[road]))
    print('Total : ', m.objVal)