import numpy as np
import pandas as pd
import os

thepath = os.getcwd()

n = 90
omax = 450000

fileArr = ["g870.asc", "rt50.asc", "t8h203.asc",  "a530875.asc", "a530874.asc", "a452374.asc", "a530872.asc",
           "a452372.asc"]
nbRowsArr = [36, 60, 81, 128, 137, 137, 137, 137]
nbColsArr = [15, 4, 48, 37, 12, 10, 18, 18]

fileArr = fileArr[:9]
nbRowsArr = nbRowsArr[:9]
nbColsArr = nbColsArr[:9]

nbBuses = np.sum(nbColsArr)
nbMonths = np.max(nbRowsArr) - 11
data = np.zeros((nbMonths, nbBuses, 3))
ino = 12
inye = 74

curbus = -1
output = np.full((nbBuses, nbMonths, 3), np.nan)
outputdiscr = np.full((nbBuses, nbMonths, 3), np.nan)
transitions = np.zeros((n, n))
themax = 0
for busType in range(len(fileArr)):
    thefile = fileArr[busType]
    nbRows = nbRowsArr[busType]
    nbCols = nbColsArr[busType]
    tmpdata = pd.read_csv(thepath + '/datafiles/' + thefile, header=None).values
    if tmpdata.shape[0] != nbRows*nbCols:
        print('Unexpected size')
        break
    tmpdata = tmpdata.reshape(nbRows, nbCols, order='F')
    print('Group =', busType, '; Nb at least one =', len(tmpdata[5, tmpdata[5, :] != 0]),
          '; Nb no repl = ', len(tmpdata[5, tmpdata[5, :] == 0]))

    for busId in range(nbCols):
        curbus = curbus + 1
        mo1stRepl = tmpdata[3, busId]
        ye1stRepl = tmpdata[4, busId]
        odo1stRep = tmpdata[5, busId]

        mo2ndRepl = tmpdata[6, busId]
        ye2ndRepl = tmpdata[7, busId]
        odo2ndRep = tmpdata[8, busId]

        moDataBegins = tmpdata[9, busId]
        yeDataBegins = tmpdata[10, busId]

        odoReadings = tmpdata[11, busId]

        wasreplacedone = [1 if 0 < odo1stRep <= data else 0 for data in tmpdata[11:nbRows, busId]]
        wasreplacetwice = [1 if 0 < odo2ndRep <= data else 0 for data in tmpdata[11:nbRows, busId]]
        howmanytimesreplaced = np.add(wasreplacedone, wasreplacetwice)

        correctedmileage = tmpdata[11:nbRows, busId] + howmanytimesreplaced * (howmanytimesreplaced - 2) * odo1stRep \
                   - 0.5 * howmanytimesreplaced * (howmanytimesreplaced - 1) * odo1stRep

        output[curbus, :(nbRows - 12), 0] = howmanytimesreplaced[1:nbRows - 11] - howmanytimesreplaced[:nbRows - 12]
        output[curbus, :(nbRows - 12), 1] = correctedmileage[:nbRows - 12]
        output[curbus, :(nbRows - 12), 2] = tmpdata[12:nbRows, busId] - tmpdata[11: nbRows - 1, busId]

        outputdiscr[curbus, :, 0] = output[curbus, :, 0]
        outputdiscr[curbus, :, 1:3] = np.ceil(n * output[curbus, :, 1:3] / omax)

        for t in range(nbRows - 13):
            i = outputdiscr[curbus, t, 1]
            j = outputdiscr[curbus, t + 1, 1]
            transitions[int(i) - 1, int(j) - 1] = transitions[int(i) - 1, int(j) - 1] + 1
