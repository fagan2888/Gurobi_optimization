import gurobipy as grb
import pandas as pd
import numpy as np
import os

# Setting up the data
thepath = os.getcwd()
filename = '/StiglerData1939.txt'
thedata = pd.read_csv(thepath + filename, sep='\t')
names = thedata.iloc[:, 0].dropna().values
names = names[:-1]
themat = thedata.iloc[:, 2:13].fillna(0).values
themat = themat[:-2, :]
intake = themat[:-1, 2:].T
allowance = themat[-1, 2:]

# Setting up the model
m = grb.Model('optimalDiet')
meal = m.addVars(names, name='meal')
m.setObjective(meal.sum(), grb.GRB.MINIMIZE)
m.addConstrs((grb.quicksum(meal[k] * intake[i, j] for j, k in enumerate(names)) >= allowance[i]
             for i in range(intake.shape[0])), name = 'c')

# Display optimal solution
m.optimize()
if m.status == grb.GRB.Status.OPTIMAL:
    total = 0
    solution = m.getAttr('x', meal)
    print('***Optimal solution***')
    for food in names:
        if solution[food] > 0:
            print(food, solution[food] * 365)
            total += solution[food] * 365
    print('Total cost (optimal) =', total)