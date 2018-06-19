import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
import os

thepath = os.getcwd()
travelmodedataset = pd.read_csv(thepath + '/travelmodedata.csv', sep=',').values

le = LabelEncoder()
lb = LabelBinarizer()
travelmodedataset[:, 1] = le.fit_transform(travelmodedataset[:, 1])
travelmodedataset[:, 2] = lb.fit_transform(travelmodedataset[:, 2]).ravel()

nobs = travelmodedataset.shape[0]
ncols = travelmodedataset.shape[1]
nind = nobs / 4
travelmodedataset = np.array(travelmodedataset, dtype=np.int32).reshape((4, int(nind), ncols), order='F')
choices = travelmodedataset[:, :, 2]
s = np.mean(choices, axis=1)
print('Market shares (air, train, bus, car)')
print(s)

Ulogit = np.log(s/s[3])
print('Systematic utilities (logit): (air, train, bus, car)')
print(Ulogit)
lamb = np.array([1/2, 1/2])

Unocar = lamb[0]*np.log(s[:3]) + (1-lamb[0]) * np.log(np.sum(s[:3]))
Ucar = lamb[1] * np.log(s[3]) + (1 - lamb[1]) * np.log(np.sum(s[3]))
Unested = np.hstack((Unocar, Ucar)) - Ucar
print('Systematic utilities (nested logit): (air, train, bus, car)')
print(Unested)

print('Choice probabilities within nocar nest (predicted vs observed): (air, train, bus)')
print(np.exp(Unested[:3] / lamb[0]) / np.sum(np.exp(Unested[:3]/lamb[0])))
print(s[:3]/np.sum(s[:3]))

print('Choice probabilites of car nest (predicted vs observed):')
print(1/(np.power(np.sum(np.exp(Unested[:3]/lamb[0])), lamb[0]) + 1))
print(s[3])