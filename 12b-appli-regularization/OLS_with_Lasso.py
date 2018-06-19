import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from glmnet import glmnet
from cvglmnet import cvglmnet
from glmnetPlot import glmnetPlot
from cvglmnetPlot import cvglmnetPlot

np.random.seed(777)
nObs = 10
nFeature = 10
nonZero = 5
data = np.array([0.37571481,-0.01561478, -0.30976885,  0.99009982,  0.39053434, -0.97859993, -0.30996829, -0.65590104,  0.89872134, -0.50161472,  0.46558062,  0.32057836,  0.16063384,  0.18956305,  0.73254298, -0.79219474,
                 -0.16338465,  0.73504565, -0.29528617, -0.22034928, -0.23907145,  0.28461128,  0.04319417, -0.64457838, -0.94001864,  0.54716325, -0.02692865,  0.11723781,  0.97919445,  0.40393124,  0.66911865,  0.45358380,
                 0.66525883, -0.33883180, -0.41364969, -0.38207253,  0.49792168, -0.44481110,  0.38350126,  0.28763969, -0.95770805, -0.43710236,  0.18016084, -0.33730981, -0.56638766,  0.97805249,  0.79719199,  0.81086278,
                 0.84948384, -0.69704737,  0.33062697, -0.28121548, -0.91892047,-0.67871307, -0.02019713, -0.69064650, -0.86333395,  0.91445373, -0.41188947, -0.95885120,  0.49158834, -0.60408428,  0.60929640, -0.98815621,
                 0.78930931, -0.87230154, -0.77552060, -0.48716068,  0.87227955, 0.22768697, -0.78353508,  0.54632708,  0.90425732, -0.17913590, -0.38544650,  0.81354321, -0.43705904,  0.33061684,  0.21419994, -0.22197407,
                 -0.23680104, -0.05085667, -0.32250169, -0.16789222, -0.53546786, -0.03741720, -0.59592697, -0.49497207,  0.46992415,  0.78034136,  0.57488618, -0.80286747, -0.53455134,  0.77346870, -0.59667691,  0.64346120,
                 -0.26849156, -0.55527494,  0.86219293,  0.86388174])
#data = np.random.uniform(1, -1, size=nObs * nFeature)
x = data.reshape((nObs, nFeature), order='F')
beta = np.hstack((list(range(1, nonZero + 1)), np.repeat(0, nFeature - nonZero)))
data2 = np.array([1.5328109, -1.2101051,  2.3302073, -0.0698569,  0.2512859,  0.3400069, -0.5278876,  0.3326885, -2.4366560, 0.8049617])
#data2 = np.random.normal(size=nObs)
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
