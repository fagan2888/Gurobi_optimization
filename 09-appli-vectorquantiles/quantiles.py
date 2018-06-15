import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gurobipy as grb
import os
import seaborn as sns
import statsmodels.formula.api as smf

t = 0.5
thePath = os.getcwd()

engeldataset = pd.read_csv(thePath + '/EngelDataset.csv', sep=',')

n = engeldataset.shape[0]
ncoldata = engeldataset.shape[1]

X0 = engeldataset.iloc[:, ncoldata - 1]

Y = engeldataset.iloc[:,0]

thedata = pd.DataFrame({'X0': X0, 'Y': Y})
ax = sns.lmplot('X0', 'Y', thedata, size=12)
plt.grid()
plt.show()

mod = smf.quantreg('Y ~ X0', thedata)
res = mod.fit(t)
print(res.summary())

X = np.column_stack((np.ones(len(X0)), X0))
k = X.shape[1]
obj = np.concatenate([np.repeat(t, n), np.repeat(1 - t, n), np.repeat(0, k)])
var = [(i, j) for i in range(n) for j in range(2)]

m = grb.Model('quantile')
quantiles = m.addVars(var, obj=obj, lb=0 ,name='quantile')
intercept = m.addVar(lb=0, name='intercept')
x0 = m.addVar(lb=0, name='x0')
m.addConstrs(quantiles[person, 0] - quantiles[person, 1] + intercept + X0[person] * x0 == Y[person] for person in range(n))

m.optimize()

if m.status == grb.GRB.Status.OPTIMAL:
    thebeta = [intercept.X, x0.X]
    print(thebeta)


def VQRTp(X, Y, U, mu, nu):
    n = Y.shape[0]
    d = Y.shape[1]
    r = X.shape[1]
    l = U.shape[0]
    if n != X.shape[0] or d != U.shape[1]:
        raise ValueError('Wrong dimensions')
    xbar = nu.dot(X)
    c = - U.dot(Y.T).T.ravel()
    mu.dot(xbar.reshape(1, 2))
    f = mu.dot(xbar.reshape(1, len(xbar)))
    varList = [(i, j) for i in range(n) for j in range(l)]

    m = grb.Model('VQRTp')
    var = m.addVars(varList, obj=c, ub=1, name='var')
    m.addConstrs(var.sum(i, '*') == nu[i] for i in range(n))
    for x in range(X.shape[1]):
        m.addConstrs(grb.quicksum(X[i, x] * var[i, k] for i in range(n)) == f[0, x] for k in range(l))

    m.optimize()
    if m.status == grb.GRB.Status.OPTIMAL:
        pivec = m.getAttr('x')
        Lvec = m.getAttr('pi')
        pi = np.array(pivec).reshape(l, -1, order='F')
        L1vec = np.array(Lvec[:n])
        L2vec = np.array(Lvec[n:n + l * r])

        psi = - L1vec.reshape(-1, 1)
        b = - L2vec.reshape(l, -1, order='F')
        val = np.trace(U.T.dot(pi).dot(Y))

        return {'pi': pi, 'psi': psi, 'b': b, 'val': val}
    else:
        print('Optimization problem with gurobi')
        return None


def ComputeBeta1D(mu, b):
    m = mu.shape[0]
    D = np.eye(m) - np.eye(m, k=-1)
    beta = np.diag((1/mu).ravel()).dot(D).dot(b)
    return (beta)


nu = np.repeat(1/n, n)
step = 0.1
Ts = np.linspace(0, 1, 11, endpoint=True)
m = len(Ts)
U = Ts.reshape(m, 1)
mu = np.repeat(1/m, m).reshape(m, 1)
d = 1
Y = Y.values.reshape(n, d)
sols = VQRTp(X, Y, U, mu, nu)
pi = sols['pi']
psi = sols['psi']
b = sols['b']
val = sols['val']
betasVQR = ComputeBeta1D(mu, b)
thebetaVQR = betasVQR[5, :]