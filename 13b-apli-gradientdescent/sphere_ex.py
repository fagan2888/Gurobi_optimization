## Gregory Dannay
## https://github.com/gdannay

import numpy as np
from scipy.optimize import minimize


def fn(x):
    return np.sum(np.square(x))


def gn(x):
    return np.multiply(x, 2)


n = 10

x0 = np.repeat(2, n)

resopt = minimize(fn, x0, method='BFGS', jac=gn)

print(resopt)
