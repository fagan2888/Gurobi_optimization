## Gregory Dannay
## https://github.com/gdannay

import numpy as np
import os
import pandas as pd
import rpy2.robjects as robjects
import gurobipy as grb
from scipy.optimize import minimize
from scipy.special import digamma

thepath = os.getcwd()
robjects.r.load(thepath + '/ChooSiowData/nSinglesv4.RData', verbose=False)
robjects.r.load(thepath + '/ChooSiowData/nMarrv4.RData', verbose=False)
robjects.r.load(thepath + '/ChooSiowData/nAvailv4.RData', verbose=False)

nbCateg = 25
nSingles = np.array(robjects.r['nSingles70n'])
marr = robjects.r['marr70nN']
nAvail = np.array(robjects.r['avail70n'])

muhatx0 = nSingles[:nbCateg, 0]
muhat0y = nSingles[nbCateg, 1]
muhatxy = np.array(marr[0])[:nbCateg, :nbCateg]
then = np.array(nAvail[0])[:nbCateg, 0]
them = np.array(nAvail[0])[:nbCateg, 1]
nbIndiv = np.sum(then) + np.sum(them)
then = then / nbIndiv
them = them / nbIndiv

nbX = len(then)
nbY = len(them)

Xs = np.array(range(1, nbCateg + 1)) + 15
Ys = np.array(range(1, nbCateg + 1)) + 15

thephi = -np.abs((np.tile(Xs, nbCateg).reshape(nbX, nbY) - np.repeat(Xs, nbCateg).reshape(nbX, nbY))) / 20


def edgeGradient(Phi, n, m, gtol=1E-15, ftol=1E-15):
    nbX = len(n)
    nbY = len(m)

    def eval_f(theU):
        theU = np.array(theU).reshape(nbX, nbY)
        theV = Phi - theU
        denomG = 1 + np.sum(np.exp(theU), axis=1)
        denomH = 1 + np.sum(np.exp(theV), axis=0)
        valG = np.sum(np.multiply(n, np.log(denomG)))
        valH = np.sum(np.multiply(m, np.log(denomH)))
        return valG + valH

    U_init = Phi / 2
    resopt = minimize(eval_f, U_init, method='L-BFGS-B', options={'ftol': ftol, 'gtol': gtol, 'maxfun': 100000})
    Usol = resopt['x'].reshape(nbCateg, nbCateg)
    mu = np.multiply(np.exp(Usol).T, (n / (1 + np.sum(np.exp(Usol), axis=1))).T).T
    mux0 = n - np.sum(mu, axis=1)
    mu0y = m - np.sum(mu, axis=0)
    val = np.sum(mu * Phi) - 2 * np.sum(mu * np.log(mu / (np.sqrt(n.reshape(-1, 1).dot(m.reshape(1, -1)))))) - \
          np.sum(mux0 * np.log(mux0 / n)) - np.sum(mu0y * np.log(mu0y / m))
    return {'mu': mu, 'mux0': mux0, 'muy0': mu0y, 'val': val, 'iter': resopt['nit']}


def simulatedLinprogr(Phi, n, m, nbDraws=1000, seed=777):
    nbX = len(n)
    nbY = len(m)
    nbI = nbX * nbDraws
    nbJ = nbY * nbDraws
    robjects.r("set.seed(" + str(seed) + ")")
    data = np.array(robjects.r("runif(" + str(nbI * nbY) + ")"))
    epsilon_iy = (digamma(1) - np.log(-np.log(data))).reshape(nbI, nbY, order='F')
    data = np.array(robjects.r("runif(" + str(nbI) + ")"))
    epsilon0_i = digamma(1) - np.log(-np.log(data))
    I_ix = np.zeros((nbI, nbX))
    for x in range(nbX):
        I_ix[range(nbDraws * x, nbDraws * (x + 1)), x] = 1
    data = np.array(robjects.r("runif(" + str(nbX * nbJ) + ")"))
    eta_xj = (digamma(1) - np.log(-np.log(data))).reshape(nbX, nbJ, order='F')
    data = np.array(robjects.r("runif(" + str(nbI) + ")"))
    eta0_j = digamma(1) - np.log(-np.log(data))
    I_yj = np.zeros((nbY, nbJ))
    for y in range(nbY):
        I_yj[y, range(nbDraws * y, nbDraws * (y + 1))] = 1
    ni = I_ix.dot(n) / nbDraws
    mj = m.dot(I_yj) / nbDraws

    width = np.max((nbX, nbY))
    varList = [(0, i) for i in range(nbI)]
    varList.extend([(1, i) for i in range(nbJ)])
    varList.extend([(2, i) for i in range(width * width)])
    obj = np.hstack((ni, mj, np.repeat(0, nbX * nbY)))
    lb = np.hstack((epsilon0_i, eta0_j, np.repeat(float('-Inf'), nbX * nbY)))
    rhs1 = epsilon_iy.T.ravel()
    rhs2 = (eta_xj + Phi.dot(I_yj)).T.ravel()

    mod = grb.Model()
    var = mod.addVars(varList, obj=obj, lb=lb, name='var')
    mod.addConstrs(var[0, i] - var[2, j * nbY + i // nbDraws] >= rhs1[j * nbI + i] for j in range(nbY) for i in range(nbI))
    mod.addConstrs(var[1, i] + var[2, j + (i // nbDraws) * nbX] >= rhs2[j + i * nbX] for i in range(nbJ) for j in range(nbX))

    mod.optimize()
    if mod.status == grb.GRB.Status.OPTIMAL:
        muiy = np.array(mod.getAttr('pi')[:nbI * nbY]).reshape(nbI, -1, order='F')
        mu = I_ix.T.dot(muiy)
        x = np.array(mod.getAttr('x'))
        val = np.sum(ni * x[:nbI]) + np.sum(mj * x[nbI:nbI + nbJ])
        mux0 = n - np.sum(mu, axis=1)
        mu0y = m - np.sum(mu, axis=0)
        return {'mu': mu, 'mux0': mux0, 'mu0y': mu0y, 'val': val}
    return None


def nodalGradient(Phi, n, m, gtol=1e-8, ftol=1e-15):
    K = np.exp(Phi / 2)
    nbX = len(n)
    nbY = len(m)
    ab_init = -np.hstack((np.log(n / 2), np.log(m / 2)))

    def eval_f(ab):
        a = ab[:nbX]
        b = ab[nbX: nbX + nbY]
        A = np.exp(-a / 2)
        B = np.exp(-b / 2)
        A2 = A * A
        B2 = B * B
        val = np.sum(n * a) + np.sum(m * b) + 2 * A.reshape(1, -1).dot(K.dot(B)) + np.sum(A2) + np.sum(B2)
        return val

    resopt = minimize(eval_f, ab_init, method='L-BFGS-B', options={'ftol': ftol, 'gtol': gtol, 'maxfun': 100000})
    absol = resopt['x']
    a = absol[:nbX]
    b = absol[nbX: nbX + nbY]
    A = np.exp(-a / 2)
    B = np.exp(-b / 2)
    mu = (A * (B * K).T).T
    mux0 = n - np.sum(mu, axis=1)
    mu0y = m - np.sum(mu, axis=0)
    val = np.sum(mu * Phi) - 2 * np.sum(mu * np.log(mu / (np.sqrt(n.reshape(-1, 1).dot(m.reshape(1, -1)))))) - \
          np.sum(mux0 * np.log(mux0 / n)) - np.sum(mu0y * np.log(mu0y / m))
    return {'mu': mu, 'mux0': mux0, 'muy0': mu0y, 'val': val, 'iter': resopt['nit']}


def ipfp(Phi, n, m, tol=1e-6):
    K = np.exp(Phi / 2)
    tK = K.T
    B = np.sqrt(m)
    iter = 0
    cont = True
    while cont:
        iter = iter + 1
        KBover2 = K.dot(B) / 2
        A = np.sqrt(n + KBover2 * KBover2) - KBover2
        tKAover2 = tK.dot(A) / 2
        B = np.sqrt(m + tKAover2 * tKAover2) - tKAover2
        discrepancy = np.max(np.abs(A * (K.dot(B) + A) - n) / n)
        if discrepancy < tol:
            cont = False
    mu = (A * (B * tK).T).T
    mux0 = n - np.sum(mu, axis=1)
    mu0y = m - np.sum(mu, axis=0)
    val = np.sum(mu * Phi) - 2 * np.sum(mu * np.log(mu / (np.sqrt(n.reshape(-1, 1).dot(m.reshape(1, -1)))))) - \
        np.sum(mux0 * np.log(mux0 / n)) - np.sum(mu0y * np.log(mu0y / m))
    return {'mu': mu, 'mux0': mux0, 'muy0': mu0y, 'val': val, 'iter': iter}


def printStats(n, m, mu, phi, lambd):
    avgAbsDiff = -np.sum(mu * phi) / np.sum(mu)
    fractionMarried = 2 * np.sum(mu) / (np.sum(n) + np.sum(m))
    print("Value of lambda", lambd)
    print("Average abolute age difference between matched partners", avgAbsDiff)
    print("Fraction of married individuals", fractionMarried)


thelambda = 1
res_edgeGradient = edgeGradient(thelambda * thephi, then, them)
res_nodalGradient = nodalGradient(thelambda*thephi, then, them)
res_simulatedLinprogr = simulatedLinprogr(thelambda*thephi, then, them, 3)
res_ipfp = ipfp(thelambda * thephi, then, them)

printStats(then, them, res_ipfp['mu'], thephi, thelambda)

print("Values returned")
print("Edge gradient  = ", res_edgeGradient['val'])
print("Nodal gradient = ", res_nodalGradient['val'])
print("IPFP           = ", res_ipfp['val'])
print("Linear progr   = ", res_simulatedLinprogr['val'])

print("Number of iterations")
print("Edge gradient  = ", res_edgeGradient['iter'])
print("Nodal gradient = ", res_nodalGradient['iter'])
print("IPFP           = ", res_ipfp['iter'])
