## Gregory Dannay
## https://github.com/gdannay

import numpy as np
from scipy.optimize import minimize
import rpy2.robjects as robjects

seed = 2018


def sigm(x):
    return 1/ (1 + np.exp(-x))


n_dim = 5
n_samp = 1000

robjects.r("set.seed(" + str(seed) + ")")
data = robjects.r("rnorm(" + str(n_dim * n_samp) + ")")
X = np.array(data).reshape(-1, n_dim, order='F')
data2 = np.array([3.563714, 3.948559, 1.963297, 2.306862, 2.294043])
data2 = robjects.r("runif(" + str(n_dim) + ",1, 4)")
theta_0 = np.array(data2)

mu = sigm(X.dot(theta_0))
y = np.zeros(n_samp)

for i in range(n_samp):
    y[i] = np.array(robjects.r("rbinom(1,1," + str(mu[i]) + ")"))


def fn(theta, y, X):
    mu = sigm(X.dot(theta))
    val = y * np.log(mu) + (1 - y) * np.log(1 - mu)
    return -np.sum(val)


def grr(theta, y, X):
    mu = sigm(X.dot(theta))
    return X.T.dot(mu - y)


x0 = np.repeat(1, n_dim)
resopt = minimize(fn, x0, method='BFGS', jac=grr, args=(y, X))
print(resopt)
