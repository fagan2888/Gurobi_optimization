import numpy as np
import gurobipy as grb

#seed = 777
#np.random.seed(seed)
nbDraws = 10
U_y = np.array([1.6, 3.2, 1.1, 0])
nbY = len(U_y)

rho = 0.5
Covar = rho * np.ones((nbY, nbY)) + (1 - rho) * np.eye(nbY)

E = np.linalg.eigh(Covar)
V = E[0]
Q = E[1]
SqrtCovar = Q.dot(np.diag(np.sqrt(V))).dot(Q.T)
data = np.array([0.48978622, -0.39854138, 0.51083632, -0.39881205, 1.63868606, 0.62127397, 0.20270430, 1.10893779, -0.20622475, -0.37896504, -0.30426166,  0.05416232, -1.88093062, -0.03375647,  2.31149476,  0.97234017,
0.96460804, -0.54413250,  0.67122325,  0.50081867, -2.03063590,  0.22775185, -0.78302514,  1.27359211, 1.43771067, 0.42700865, -1.74445653, -0.02531606, -1.48832176, -0.54157616,  0.66131292,  0.85834207,
1.25166848, -1.21470298,  1.52315292, -1.23598286,  1.66585308, -0.50358433, -0.57847871,  0.27176845])
#data = np.random.normal(size=nbDraws*nbY)
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

