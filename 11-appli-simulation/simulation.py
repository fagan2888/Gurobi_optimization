## Gregory Dannay
## https://github.com/gdannay

import numpy as np
import gurobipy as grb
import rpy2.robjects as robjects

seed = 777
nbDraws = 10
U_y = np.array([1.6, 3.2, 1.1, 0])
nbY = len(U_y)

rho = 0.5
Covar = rho * np.ones((nbY, nbY)) + (1 - rho) * np.eye(nbY)

E = np.linalg.eigh(Covar)
V = E[0]
Q = E[1]
SqrtCovar = Q.dot(np.diag(np.sqrt(V))).dot(Q.T)
robjects.r("set.seed(" + str(seed) + ")")
data = robjects.r("rnorm(" + str(nbDraws * nbY) + ")")
data = np.array(data)
epsilon_iy = data.reshape(-1, nbY, order='F').dot(SqrtCovar)
u_iy = epsilon_iy + U_y

ui = np.max(u_iy, axis=1)
s_y = np.sum((u_iy.T - ui).T == 0, axis=0) / nbDraws

opt_assign = [(i, j) for i in range(nbY) for j in range(nbDraws)]

m = grb.Model()
vars = m.addVars(opt_assign, obj=epsilon_iy.T.ravel(), name='vars')
m.ModelSense = -1
m.addConstrs(vars.sum('*', i) == 1/nbDraws for i in range(nbDraws))
m.addConstrs(vars.sum(i, '*') == s_y[i] for i in range(nbY))

m.optimize()
if m.Status == grb.GRB.Status.OPTIMAL:
    pi = m.getAttr('pi')
    Uhat_y = -np.subtract(pi[nbDraws:nbY+nbDraws], pi[nbY + nbDraws - 1])
    print('U_y (true and recovered)')
    print(U_y)
    print(Uhat_y)

