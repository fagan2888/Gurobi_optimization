## Gregory Dannay
## https://github.com/gdannay

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from glmnet import glmnet
from cvglmnet import cvglmnet
from glmnetPlot import glmnetPlot
from cvglmnetPlot import cvglmnetPlot
import rpy2.robjects as robjects


seed = 777
nObs = 10
nFeature = 10
nonZero = 5
robjects.r("set.seed(" + str(seed) + ")")
data = robjects.r("runif(" + str(nObs * nFeature) + ", -1, 1)")
data = np.array(data)
x = data.reshape((nObs, nFeature), order='F')
beta = np.hstack((list(range(1, nonZero + 1)), np.repeat(0, nFeature - nonZero)))
data2 = robjects.r("rnorm(" + str(nObs) + ")")
data2 = np.array(data2)
eps = 0.5 * data2
y = x.dot(beta) + eps
plt.plot(y, marker='o', linestyle='None')
plt.show()

LR = LinearRegression(fit_intercept=False)
LR.fit(x, y)
print(LR.coef_)

fit = glmnet(x=x.copy(), y=y.copy(), alpha=1, intr=False)
glmnetPlot(fit)

cvfit = cvglmnet(x=x.copy(), y=y.copy(), nfolds=5, alpha=1, intr=False)
cvglmnetPlot(cvfit)

LR = LinearRegression(fit_intercept=False)
LR.fit(x[:10], y)
print(LR.coef_)
