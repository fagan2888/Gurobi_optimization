import numpy as np
import gurobipy as grb
import os
import pandas as pd
import time

syntheticData = False
doGurobi = False
doIPFP1 = False
doIPFP1bis = False
doIPFP2 = True

tol = 1E-9
maxiter = 1000000
sigma = 0.1

if syntheticData:
    seed = 777
    nbX = 10
    nbY = 8
    np.random.seed(seed)
    #Phi = np.random.uniform(size=(nbX * nbY)).reshape((nbX, nbY))
    Phi = np.array((0.68785741, 0.7327903, 0.38046427, 0.8345593, 0.02114597, 0.66531348, 0.7457941709, 0.1082325,
                    0.49219261, 0.6602892, 0.64230564, 0.726791, 0.28144882, 0.35939226, 0.197957861, 0.7731635, 0.34511557,
                    0.5803169, 0.52159708, 0.8326294, 0.59008042, 0.04053976, 0.804648202, 0.9521287, 0.99504991, 0.5947815,
                    0.17771081, 0.3305841, 0.33134509, 0.16064347, 0.005921893, 0.4104320, 0.69526717, 0.8662715, 0.02999068,
                    0.2931752, 0.21680617, 0.48990144, 0.894654655, 0.3072768, 0.01070004, 0.1039026, 0.77358162, 0.3089637,
                    0.98902624, 0.15467675, 0.063849231, 0.9067716, 0.34501585, 0.4183077, 0.48653567, 0.7489608, 0.89859600,
                    0.06833302, 0.112239698, 0.2814705, 0.17204948, 0.8675228, 0.55861891, 0.2775945, 0.90543139, 0.95722686, 0.256419660,
                    0.6653084, 0.94936067, 0.3523569, 0.98959723, 0.6917506, 0.92474192, 0.29405526, 0.936139773, 0.6071000, 0.24919264,
                    0.3898254, 0.70196562, 0.6438198, 0.15147631, 0.02057440, 0.613843485, 0.3890130)).reshape((nbX, nbY))
    p = np.repeat(1/nbX, nbX)
    q = np.repeat(1/nbY, nbY)
else:
    thePath = os.getcwd()
    data = pd.read_csv(thePath + '/affinitymatrix.csv', sep=',')
    nbcar = 10
    A = data.iloc[:nbcar, 1:nbcar + 1].values
    data = pd.read_csv(thePath + '/Xvals.csv', sep=',')
    Xvals = data.iloc[:,:nbcar].values
    data = pd.read_csv(thePath + '/Yvals.csv', sep=',')
    Yvals = data.iloc[:,:nbcar].values
    Xvals = (Xvals - np.mean(Xvals, axis=0)) / np.std(Xvals, axis=0, ddof=1)
    Yvals = (Yvals - np.mean(Yvals, axis=0)) / np.std(Yvals, axis=0, ddof=1)
    nobs = Xvals.shape[0]
    Phi = Xvals.dot(A).dot(Yvals.T)
    p = np.repeat(1/nobs, nobs)
    q = np.repeat(1/nobs, nobs)
    nbX = len(p)
    nbY = len(q)

nrow = min(8, nbX)
ncol = min(8, nbY)

if doGurobi:
    coupleList = [(i, j) for i in range(nbY) for j in range(nbX)]
    ptm = time.time()
    m = grb.Model('marriage')
    couple = m.addVars(coupleList, obj=Phi.T.ravel(), name='couple')
    m.ModelSense = -1
    m.addConstrs((couple.sum('*', women) == p[women] for women in range(nbX)))
    m.addConstrs((couple.sum(men, '*') == q[men] for men in range(nbY)))
    m.optimize()
    diff = time.time() - ptm
    print('Time elapsed (Gurobi) = ', diff, 's.')
    if m.status == grb.GRB.Status.OPTIMAL:
        val_gurobi = m.objval
        pi = m.getAttr(grb.GRB.Attr.Pi)
        u_gurobi = pi[:nbX]
        v_gurobi = pi[nbX:nbX + nbY]
        print("Value of the problem (Gurobi) = ", val_gurobi)
        print(np.subtract(u_gurobi[:nrow], u_gurobi[nrow - 1]))
        print(np.add(v_gurobi[:ncol], u_gurobi[nrow - 1]))
        print('*************************')

if doIPFP1:
   ptm = time.time()
   iter = 0

   K = np.exp(Phi/sigma)
   B = np.repeat(1, nbY)
   error = tol + 1
   while error > tol and iter < maxiter:
       A = p / K.dot(B)
       KA = A.dot(K)
       error = np.max(KA * B / q - 1)
       B = q / KA
       iter = iter + 1
   u = - sigma * np.log(A)
   v = - sigma * np.log(B)
   pi = (K.T * A).T * np.tile(B, nbX).reshape(nbX, nbY)
   val = np.sum(pi * Phi) - sigma * np.sum(pi * np.log(pi))
   end = time.time() - ptm
   if iter >= maxiter:
       print('Maximum number of iterations reached in IPFP1.')
   else:
       print('IPFP1 converged in ', iter, ' steps and ', end, 's.')
       print('Value of the problem (IPFP1) = ', val)
       print('Sum(pi*Phi) (IPFP1) = ', np.sum(pi*Phi))
       print(np.subtract(u[:nrow], u[nrow - 1]))
       print(np.add(v[:ncol], u[nrow - 1]))
   print('*************************')

if doIPFP1bis:
    ptm = time.time()
    iter = 0
    v = np.repeat(0, nbY)
    mu = - sigma * np.log(p)
    nu = - sigma * np.log(q)
    error = tol + 1
    while error > tol and iter < maxiter:
        u = mu + sigma * np.log(np.sum(np.exp((Phi - np.tile(v, nbX).reshape(nbX, nbY))/sigma), axis=1))
        KA = np.sum(np.exp((Phi.T - u).T / sigma), axis=0)
        error = np.max(np.abs(KA*np.exp(-v / sigma) / q - 1))
        v = nu + sigma * np.log(KA)
        iter = iter + 1
    pi = np.exp(((Phi.T - u).T - np.tile(v, nbX).reshape(nbX, nbY)) / sigma)
    val = np.sum(pi * Phi) - sigma * np.sum((pi * np.log(pi))[pi != 0])
    end = time.time() - ptm

    if iter >= maxiter:
        print('Maximum number of iterations reached in IPFP1.')
    else:
        print('IPFP1bis converged in ', iter, ' steps and ', end, 's.')
        print('Value of the problem (IPFP1bis) = ', val)
        print('Sum(pi*Phi) (IPFP1bis) = ', np.sum(pi*Phi))
        print(np.subtract(u[:nrow], u[nrow - 1]))
        print(np.add(v[:ncol], u[nrow - 1]))
    print('*************************')

if (doIPFP2):
    ptm = time.time()
    iter = 0
    v = np.repeat(0, nbY)
    mu = - sigma * np.log(p)
    nu = - sigma * np.log(q)
    error = tol + 1
    uprec = float('-inf')
    while error > tol and iter < maxiter:
        vstar = np.max((Phi.T - v).T, axis=1)
        u = mu + vstar + sigma * np.log(np.sum(np.exp(((Phi - np.tile(v, nbX).reshape(nbX, nbY)).T - vstar).T/sigma), axis=1))
        error = np.max(np.abs(u - uprec))
        uprec = u
        ustar = np.max((Phi.T - u).T, axis=0)
        v = nu + ustar + sigma * np.log(np.sum(np.exp(((Phi.T - u).T - np.tile(ustar, nbX).reshape(nbX, nbY))/sigma), axis=0))
        iter = iter + 1
    pi = np.exp(((Phi.T - u).T - np.tile(v, nbX).reshape(nbX, nbY)) / sigma)
    val = np.sum(pi * Phi) - sigma * np.sum(pi * np.log(pi))
    end = time.time() - ptm
    if iter >= maxiter:
        print('Maximum number of iterations reached in IPFP2.')
    else:
        print('IPFP2 converged in ', iter, ' steps and ', end, 's.')
        print('Value of the problem (IPFP2) = ', val)
        print('Sum(pi*Phi) (IPFP2) = ', np.sum(pi*Phi))
        print(np.subtract(u[:nrow], u[nrow - 1]))
        print(np.add(v[:ncol], u[nrow - 1]))
    print('*************************')
