import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib import colors as mcolors
from matplotlib import cm
import seaborn as sns
from scipy.optimize import curve_fit

def getCrossChannels(histo, startChannel, detector=0):
    len = histo.shape[1]
    out = np.zeros((len, histo.shape[3]), dtype=np.int32)
    for i in range(len):
        out[i] = histo[detector][startChannel][i]
    return out


def getTimeSlice(histos, startingDet, gOne):
    timeScale = 2.5 * 1e-9
    # do 2d Plot of spatio temporal regime
    bins = np.arange(-40e-9, 50e-9, timeScale)
    histoNow, histAcc = histos
    histoMaster = getCrossChannels(histAcc, startingDet, 0)
    histoSlave = getCrossChannels(histAcc, startingDet, 1)
    binDurationMaster = 23.616e-12
    binDurationSlave = 22.22e-12
    offsetMaster = 1559
    offsetSlave = 1591
    mergedImag = np.zeros((histoMaster.shape[0], len(bins) - 1), dtype=np.float64)
    for i in range(histoMaster.shape[0]):
        singleHistoMaster = histoMaster[i]
        singleHistoSlave = histoSlave[i]
        valsNeeded = []
        for j in range(4096):
            timeMaster = float((j - offsetMaster)) * binDurationMaster
            timeSlave = float((j - offsetSlave)) * binDurationSlave
            # append for master
            valsNeeded.append([timeMaster] * singleHistoMaster[j])
            # append for slave
            valsNeeded.append([timeSlave] * singleHistoSlave[j])
        # ravel
        flat_list = [item for sublist in valsNeeded for item in sublist]
        ravelled = np.array(flat_list)
        histo = np.histogram(ravelled, bins)
        # add column to imag
        mergedImag[i] = histo[0]
    # got histos, create imag
    # jup
    # copy normalization one
    normedImag = mergedImag
    # transpose imag
    mergedImag = mergedImag.T
    # normalize
    imagBinnedMaster, imagBinnedSlave = gOne
    totaloBinned = np.add(imagBinnedSlave[1], imagBinnedMaster[1])
    totaloBinned = totaloBinned / np.sum(totaloBinned)
    newNormalized = np.zeros_like(normedImag)
    errorsRelative = np.zeros(normedImag.shape[0], dtype=np.float64)
    zeroBin = int(mergedImag.shape[0] / 2) - 1
    for i in range(normedImag.shape[0]):
        if (normedImag[i][zeroBin] != 0):
            errorsRelative[i] = (np.sqrt(normedImag[i][zeroBin]) / normedImag[i][zeroBin])
        newNormalized[i] = normedImag[i] / (totaloBinned[i] * totaloBinned[startingDet])
    finishedNormalied = newNormalized.T
    perBin = float(np.sum(mergedImag)) / float(mergedImag.shape[0] * mergedImag.shape[1])

    return finishedNormalied

histos = np.load("AllDAta.bin.npy")
gOne = np.load("alldataG1.npy")
gOneTot = np.load("GOneTot.npy")

for i in range(histos.shape[2]):
    temporalStart = getTimeSlice(histos, i, gOne)
    #temporalCut = temporalStart[8:17, 35:60]
    temporalCut = temporalStart
    temporalCut = np.divide(temporalCut, np.amax(temporalCut))

    ys = temporalCut[int(temporalCut.shape[0] / 2) - 1]
    plt.plot(ys)
    plt.title("starting det: " + str(i))
    plt.show()

print("finish")