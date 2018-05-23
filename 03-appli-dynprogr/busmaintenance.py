import numpy as np
import gurobipy as grb
from scipy import sparse

# Setting up the parameters
nbX = 10
nbT = 40
nbY = 2

# Creating the data
xtList = [(i, j) for i in range(nbX) for j in range(nbT)]

b_xt = np.zeros(nbX * nbT)
b_xt[:nbX] = 1

beta = 0.9
beta_t = np.power(beta, range(1, nbT + 1))
overhaultCost = 8000
maintCost = lambda x: x*np.multiply(5, 100)
u_x1 = np.hstack((maintCost(range(1, nbX)), 8000))
u_x1t = -np.kron(beta_t, u_x1)
u_x1t = np.reshape(u_x1t, (nbT, nbX))
u_x2t = beta_t * overhaultCost * -1

P = np.zeros((nbX * nbY, nbX))
P[range(0, nbX), range(0, nbX)] = 0.75
P[np.hstack((nbX - 1, range(0, nbX - 1))), range(0, nbX)] = 0.25
P[range(nbX, nbX * nbY), 0] = 1

# Setting up the model
m = grb.Model('busmaintenance')

xt = m.addVars(xtList, obj=b_xt, name='xt', lb=float('-inf'))

m.addConstrs((xt[state, time] - 0.75*xt[state, time + 1] - 0.25*xt[state + 1, time + 1] >= u_x1t[time, state]
              for state in range(nbX - 1) for time in range(nbT - 1)))
m.addConstrs((xt[state, nbT - 1] >= u_x1t[nbT - 1, state] for state in range(nbX)))
m.addConstrs((xt[nbX - 1, time] - 0.75*xt[nbX - 1, time + 1] - 0.25*xt[0, time + 1] >= u_x1t[time, nbX - 1] for time in range(nbT - 1)))
m.addConstrs((xt[state, time] - xt[0, time + 1] >= u_x2t[time]
              for state in range(nbX) for time in range(nbT - 1)))
m.addConstrs((xt[state, nbT - 1] >= u_x2t[nbT - 1] for state in range(nbX)))

# Print the Gurobi solution
m.optimize()
if m.status == grb.GRB.Status.OPTIMAL:
    solution = m.getAttr('x', xt)
    for state in range(nbX):
        print(solution[(state, 0)])


# Backward induction
U_x_t = np.zeros((nbX,nbT))
contVals = np.vstack((u_x1t[nbT - 1, :], np.repeat(u_x2t[nbT - 1], nbX))).max(0)
U_x_t[:, nbT - 1] = contVals

for t in range(nbT-2, -1, -1):
    myopic = np.vstack((u_x1t[t, :], np.repeat(u_x2t[t], nbX)))
    EcontVals = np.dot(P, contVals).reshape([nbY, nbX])
    contVals = myopic + EcontVals
    contVals = contVals.max(0)
    U_x_t[:, t] = contVals

# Print the backward induction solution
print('\n')
print(U_x_t[:, 0])