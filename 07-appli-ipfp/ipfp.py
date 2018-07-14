## Gregory Dannay
## https://github.com/gdannay

import numpy as np
import gurobipy as grb
import os
import pandas as pd
import time
import rpy2.robjects as robjects

syntheticData = True
doGurobi = True
doIPFP1 = True
doIPFP1bis = True
doIPFP2 = True

tol = 1E-9
maxite = 1000000
sigma = 0.1

if syntheticData:
    seed = 777
    nbX = 10
    nbY = 8
    np.random.seed(seed)
    robjects.r("set.seed(" + str(seed) + ")")
    data = robjects.r("runif(" + str(nbX * nbY) + ")")
    Phi = np.array(data).reshape(nbX, nbY, order='F')
    p = np.repeat(1/nbX, nbX)
    q = np.repeat(1/nbY, nbY)
else:
    thePath = os.getcwd()
    data = pd.read_csv(thePath + '/affinitymatrix.csv', sep=',')
    nbcar = 10
    A = data.iloc[:nbcar, 1:nbcar + 1].values
    data = pd.read_csv(thePath + '/Xvals.csv', sep=',')
    Xvals = data.iloc[:, :nbcar].values
    data = pd.read_csv(thePath + '/Yvals.csv', sep=',')
    Yvals = data.iloc[:, :nbcar].values
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
   ite = 0

   K = np.exp(Phi/sigma)
   B = np.repeat(1, nbY)
   error = tol + 1
   while error > tol and ite < maxite:
       A = p / K.dot(B)
       KA = A.dot(K)
       error = np.max(KA * B / q - 1)
       B = q / KA
       ite = ite + 1
   u = - sigma * np.log(A)
   v = - sigma * np.log(B)
   pi = (K.T * A).T * np.tile(B, nbX).reshape(nbX, nbY)
   val = np.sum(pi * Phi) - sigma * np.sum(pi * np.log(pi))
   end = time.time() - ptm
   if ite >= maxite:
       print('Maximum number of iteations reached in IPFP1.')
   else:
       print('IPFP1 converged in ', ite, ' steps and ', end, 's.')
       print('Value of the problem (IPFP1) = ', val)
       print('Sum(pi*Phi) (IPFP1) = ', np.sum(pi*Phi))
       print(np.subtract(u[:nrow], u[nrow - 1]))
       print(np.add(v[:ncol], u[nrow - 1]))
   print('*************************')

if doIPFP1bis:
    ptm = time.time()
    ite = 0
    v = np.repeat(0, nbY)
    mu = - sigma * np.log(p)
    nu = - sigma * np.log(q)
    error = tol + 1
    while error > tol and ite < maxite:
        u = mu + sigma * np.log(np.sum(np.exp((Phi - np.tile(v, nbX).reshape(nbX, nbY))/sigma), axis=1))
        KA = np.sum(np.exp((Phi.T - u).T / sigma), axis=0)
        error = np.max(np.abs(KA*np.exp(-v / sigma) / q - 1))
        v = nu + sigma * np.log(KA)
        ite = ite + 1
    pi = np.exp(((Phi.T - u).T - np.tile(v, nbX).reshape(nbX, nbY)) / sigma)
    val = np.sum(pi * Phi) - sigma * np.sum((pi * np.log(pi))[pi != 0])
    end = time.time() - ptm

    if ite >= maxite:
        print('Maximum number of iteations reached in IPFP1.')
    else:
        print('IPFP1bis converged in ', ite, ' steps and ', end, 's.')
        print('Value of the problem (IPFP1bis) = ', val)
        print('Sum(pi*Phi) (IPFP1bis) = ', np.sum(pi*Phi))
        print(np.subtract(u[:nrow], u[nrow - 1]))
        print(np.add(v[:ncol], u[nrow - 1]))
    print('*************************')

if (doIPFP2):
    ptm = time.time()
    ite = 0
    v = np.repeat(0, nbY)
    mu = - sigma * np.log(p)
    nu = - sigma * np.log(q)
    error = tol + 1
    uprec = float('-inf')
    while error > tol and ite < maxite:
        vstar = np.max(Phi - v, axis=1)
        u = mu + vstar + sigma * np.log(np.sum(np.exp(((Phi - np.tile(v, nbX).reshape(nbX, nbY)).T - vstar).T/sigma), axis=1))
        error = np.max(np.abs(u - uprec))
        uprec = u
        ustar = np.max((Phi.T - u).T, axis=0)
        v = nu + ustar + sigma * np.log(np.sum(np.exp(((Phi.T - u).T - np.tile(ustar, nbX).reshape(nbX, nbY))/sigma), axis=0))
        ite = ite + 1
    pi = np.exp(((Phi.T - u).T - np.tile(v, nbX).reshape(nbX, nbY)) / sigma)
    val = np.sum(pi * Phi) - sigma * np.sum(pi * np.log(pi))
    end = time.time() - ptm
    if ite >= maxite:
        print('Maximum number of iteations reached in IPFP2.')
    else:
        print('IPFP2 converged in ', ite, ' steps and ', end, 's.')
        print('Value of the problem (IPFP2) = ', val)
        print('Sum(pi*Phi) (IPFP2) = ', np.sum(pi*Phi))
        print(np.subtract(u[:nrow], u[nrow - 1]))
        print(np.add(v[:ncol], u[nrow - 1]))
    print('*************************')
