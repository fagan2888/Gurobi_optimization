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
travelmodedataset = np.array(travelmodedataset).reshape((4, int(nind), ncols), order='F')
choices = travelmodedataset[:, :, 2]
