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


sns.set()

histos = np.load("AllDAta.bin.npy")
gOne = np.load("alldataG1.npy")
gOneTot = np.load("GOneTot.npy")

imagBinnedMaster, imagBinnedSlave = gOne
totaloBinned = np.add(imagBinnedSlave[1], imagBinnedMaster[1])
totaloBinned = totaloBinned / np.sum(totaloBinned)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.view_init(elev=45., azim=280.)

dummyData = getTimeSlice(histos, 47, gOne)
finishedCut = dummyData[:,30:60]

xs = np.array(range(finishedCut.shape[1]), dtype=np.float)
verts = []
zslook = np.array([41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55], dtype=np.float)
#zslook = np.flip(zslook)
zs = range(len(zslook))
for z in zs:
    temporalStart = getTimeSlice(histos, int(zslook[int(z)]), gOne)

    meanval = np.mean(temporalStart[2:6])

    temporalCut = temporalStart[:,30:60]
    temporalCut = np.divide(temporalCut, np.amax(meanval))

    ys = temporalCut[int(temporalCut.shape[0]/2)-1]
    ys[0], ys[-1] = 0, 0
    verts.append(list(zip(xs, ys)))

facecolors = [cm.jet(x) for x in np.linspace(0., 1., num=len(zs))]
black = (0,0,0,0.7)
poly = PolyCollection(verts, facecolors=facecolors, edgecolors=(black,))
poly.set_alpha(0.7)
ax.add_collection3d(poly, zs=zs, zdir='y')

ax.set_xlabel('X')
ax.set_xlim3d(0, finishedCut.shape[1])
ax.set_ylabel('Start Det')
ax.set_ylim3d(0, len(zs))
ax.set_zlabel(r'g^2')
ax.set_zlim3d(0, 1.6)

#plt.show()
plt.savefig("rollingPhase.pdf")

print("finish")