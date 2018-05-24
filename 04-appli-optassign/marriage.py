import pandas as pd
import numpy as np
import gurobipy as grb
import os

# Setting up the data
thepath = os.getcwd()
nbcar = 10

A = pd.read_csv(thepath + "/affinitymatrix.csv", sep=',').iloc[0:nbcar, 1:nbcar + 1].values
Xvals = pd.read_csv(thepath + "/Xvals.csv", sep=',').iloc[:, :nbcar + 1].values
Yvals = pd.read_csv(thepath + "/Yvals.csv", sep=',').iloc[:, :nbcar + 1].values

Xvals = (Xvals - np.mean(Xvals, axis=0)) / np.std(Xvals, axis=0, ddof=1)
Yvals = (Yvals - np.mean(Yvals, axis=0)) / np.std(Yvals, axis=0, ddof=1)
nobs = Xvals.shape[0]

Phi = Xvals.dot(A).dot(Yvals.T).T
c = Phi.ravel()

# Creating the list of variable
coupleList = [(i, j) for i in range(nobs) for j in range(nobs)]

# Creating the model
m = grb.Model('marriage')
couple = m.addVars(coupleList, obj=c, name='couple')
m.ModelSense = -1
m.addConstrs((couple.sum('*', women) == 1 for women in range(nobs)))
m.addConstrs((couple.sum(men, '*') == 1 for men in range(nobs)))

m.optimize()

# Print the solution
if m.status == grb.GRB.Status.OPTIMAL:
    print("Value of the problem (Gurobi) =", m.objval)
    pi = m.getAttr(grb.GRB.Attr.Pi)
    u = pi[:nobs]
    v = pi[nobs:]
    print(u[:10])
    print(v[:10])
