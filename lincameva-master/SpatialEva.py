import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import interpolation

from photonscore.python.flim import read
from photonscore.python.histogram import histogram
from photonscore.python.vault import Vault

from glob import glob

#from bokeh.models import HoverTool
#from bokeh.io import output_notebook

import time

from Classes.EventPreProcessor import EventPreProcessor
from Classes.EinsteinCorrection import EinsteinCorrection

def getSliceIndex(input):
    index = (input - 450)//43
    index = max(0, index)
    index = min(71, index)
    return index

starttime = time.time()

data = Vault(r"E:\RawData\SpatialIonsMainz\Measurement\01_09_2018_FirstMeas\17uW53.photons")
indexTacNonZero = np.nonzero(data.data.master.photons.tac[:])
xesM = data.data.master.photons.x[:][indexTacNonZero]
yesM = data.data.master.photons.y[:][indexTacNonZero]
aatM = data.data.master.photons.aat[:][indexTacNonZero]
tacM = data.data.master.photons.tac[:][indexTacNonZero]
chanM = np.zeros(len(xesM))
masterVals = np.array([
    xesM,
    yesM,
    aatM,
    tacM,
    chanM
]).T
indexTacNonZero = np.nonzero(data.data.slave.photons.tac[:])
xesS = data.data.slave.photons.x[:][indexTacNonZero]
yesS = data.data.slave.photons.y[:][indexTacNonZero]
aatS = data.data.slave.photons.aat[:][indexTacNonZero]
tacS = data.data.slave.photons.tac[:][indexTacNonZero]
chanS = np.ones(len(xesS))
slaveVals = np.array([
    xesS,
    yesS,
    aatS,
    tacS,
    chanS
]).T

totalVals = np.concatenate((masterVals, slaveVals))
totalVals = totalVals[np.argsort(totalVals[:, 2])]
diffs = np.diff(totalVals.T[4])
id_pos = np.where(diffs == 1)[0]
id_pos_sec = id_pos + 1
newValsX_pos = []
newValsY_pos = []
index_pos = []
index_sec_pos = []

for i in id_pos:
    sum = totalVals[i][3] + totalVals[i+1][3]
    if (np.abs(sum - 3150) < 100):
        index_pos.append(i)
        index_sec_pos.append(i+1)
        newValsX_pos.append(totalVals[i][3])
        newValsY_pos.append(totalVals[i + 1][3])

plt.plot(np.array(newValsX_pos), np.array(newValsY_pos), "o")
plt.title("fine pos")
#plt.show()

id_neg = np.where(diffs == -1)[0]
id_neg_sec = id_neg + 1
newValsX_neg = []
newValsY_neg = []
index_neg = []
index_sec_neg = []

for i in id_neg:
    sum = totalVals[i][3] + totalVals[i+1][3]
    if (np.abs(sum - 3150) < 55):
        index_neg.append(i)
        index_sec_neg.append(i+1)
        newValsX_neg.append(totalVals[i][3])
        newValsY_neg.append(totalVals[i + 1][3])

plt.plot(np.array(newValsX_neg), np.array(newValsY_neg), "o")
plt.title("fine neg")
#plt.show()

#selection of double events complete
#create 2 times 72 histos to bin each slice
histos = np.zeros((2,72,72,4096), dtype=np.int64)

rotAngleM = (-29.72883115886924 / 180.) * np.pi
rotMatrixM = np.array([
    [np.cos(rotAngleM), -1 * np.sin(rotAngleM)],
    [np.sin(rotAngleM), np.cos(rotAngleM)]
])
rotAngleS = (-5.19673564264405/180.)*np.pi
rotMatrixS = np.array([
    [np.cos(rotAngleS), -1*np.sin(rotAngleS)],
    [np.sin(rotAngleS), np.cos(rotAngleS)]
])
#translate
translateX = -69
translateY = -50
#now process neg an pos vals
sameSlice = 0
for element in np.concatenate((index_pos, index_neg)):
    firstElm = totalVals[element]
    secondElm = totalVals[element + 1]
    #create rotation and translation
    if firstElm[4] == 0:
        #Master
        coordVec = np.array([firstElm[0] - 2048, firstElm[1] - 2048])
        rotatedMasterCoords = rotMatrixM.dot(coordVec).T + 2048
        firstElm[0] = rotatedMasterCoords[0]
        firstElm[1] = rotatedMasterCoords[1]
    else:
        #slave
        coordVec = np.array([firstElm[0] - 2048, firstElm[1] - 2048])
        rotatedMasterCoords = rotMatrixS.dot(coordVec).T + 2048
        firstElm[0] = rotatedMasterCoords[0] + translateX
        firstElm[1] = rotatedMasterCoords[1] + translateY

    if secondElm[4] == 0:
        #Master
        # Master
        coordVec = np.array([secondElm[0] - 2048, secondElm[1] - 2048])
        rotatedMasterCoords = rotMatrixM.dot(coordVec).T + 2048
        secondElm[0] = rotatedMasterCoords[0]
        secondElm[1] = rotatedMasterCoords[1]
    else:
        #Slave
        coordVec = np.array([secondElm[0] - 2048, secondElm[1] - 2048])
        rotatedMasterCoords = rotMatrixS.dot(coordVec).T + 2048
        secondElm[0] = rotatedMasterCoords[0] + translateX
        secondElm[1] = rotatedMasterCoords[1] + translateY

    firstSlice = getSliceIndex(firstElm[0])
    secondSlice = getSliceIndex(secondElm[0])
    #check for same slice
    sameSlice += 1
    #append aat on histo depending on master or slave
    histos[int(firstElm[4])][int(firstSlice)][int(secondSlice)][int(firstElm[3])] += 1
    histos[int(secondElm[4])][int(secondSlice)][int(secondSlice)][int(secondElm[3])] += 1


print("Correlation completed, took " + str(time.time() - starttime) + " seconds")
print(str(sameSlice) + " Same Slices")

print("finish")