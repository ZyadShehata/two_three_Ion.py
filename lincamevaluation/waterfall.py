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


sns.set()

histos = np.load("AllDAta.bin.npy")
gOne = np.load("alldataG1.npy")

gOneTot = np.load("GOneTot.npy")
plt.axis('off')
plt.grid(False)
plt.imshow(gOneTot[1])
plt.title("g1 tot")
plt.show()

timeScale = 2.5*1e-9
startingDet = 47



#do 2d Plot of spatio temporal regime
bins = np.arange(-40e-9, 50e-9, timeScale)
histoNow, histAcc = histos
histoMaster = getCrossChannels(histAcc, startingDet,0)
histoSlave = getCrossChannels(histAcc, startingDet,1)
binDurationMaster = 23.616e-12
binDurationSlave = 22.22e-12
offsetMaster = 1559
offsetSlave = 1591
mergedImag = np.zeros((histoMaster.shape[0], len(bins)-1), dtype=np.float64)
for i in range(histoMaster.shape[0]):
    singleHistoMaster = histoMaster[i]
    singleHistoSlave = histoSlave[i]
    valsNeeded = []
    for j in range(4096):
        timeMaster = float((j - offsetMaster)) * binDurationMaster
        timeSlave = float((j - offsetSlave)) * binDurationSlave
        #append for master
        valsNeeded.append([timeMaster] * singleHistoMaster[j])
        #append for slave
        valsNeeded.append([timeSlave] * singleHistoSlave[j])
    #ravel
    flat_list = [item for sublist in valsNeeded for item in sublist]
    ravelled = np.array(flat_list)
    histo = np.histogram(ravelled, bins)
    #add column to imag
    mergedImag[i] = histo[0]
#got histos, create imag
#jup
# copy normalization one
normedImag = mergedImag
#transpose imag
mergedImag = mergedImag.T
#normalize
imagBinnedMaster, imagBinnedSlave = gOne
totaloBinned = np.add(imagBinnedSlave[1], imagBinnedMaster[1])
totaloBinned = totaloBinned / np.sum(totaloBinned)
newNormalized = np.zeros_like(normedImag)
errorsRelative = np.zeros(normedImag.shape[0] , dtype=np.float64)
zeroBin = int(mergedImag.shape[0]/2)-1
for i in range(normedImag.shape[0]):
    if (normedImag[i][zeroBin] != 0):
        errorsRelative[i] = (np.sqrt(normedImag[i][zeroBin]) / normedImag[i][zeroBin])
    newNormalized[i] = normedImag[i] / (totaloBinned[i]*totaloBinned[startingDet])
finishedNormalied = newNormalized.T
perBin = float(np.sum(mergedImag)) / float(mergedImag.shape[0] * mergedImag.shape[1])


#plot stuff

plt.grid(False)
#plt.imshow(mergedImag)
#plt.show()
plt.clf()
plt.grid(False)
plt.axis('off')
plt.title(r"$g^{(2)}\left( \vec{x}_1, \vec{x}_2, \tau \right)$")
plt.imshow(finishedNormalied)
#plt.show()

finishedCut = finishedNormalied[2:17,33:56]
finishedCut = finishedNormalied[8:17,35:60]
finishedCut = np.divide(finishedCut, np.amax(finishedCut))
finishedCut = np.flip(finishedCut, 0)

plt.clf()
plt.imshow(finishedCut)
plt.show()
plt.clf()

fig = plt.figure()
ax = fig.gca(projection='3d')

def cc(arg):
    return mcolors.to_rgba(arg, alpha=0.6)

xs = np.array(range(finishedCut.shape[1]))
verts = []
zs = np.array(range(finishedCut.shape[0]))
for z in zs:
    ys = finishedCut[z]
    ys[0], ys[-1] = 0, 0
    verts.append(list(zip(xs, ys)))

facecolors = [cm.jet(x) for x in np.random.rand(finishedCut.shape[0])]
black = (0,0,0,0.7)
poly = PolyCollection(verts, facecolors=facecolors, edgecolors=(black,))
poly.set_alpha(0.7)
ax.add_collection3d(poly, zs=zs, zdir='y')

ax.set_xlabel('X')
ax.set_xlim3d(0, finishedCut.shape[1])
ax.set_ylabel('Y')
ax.set_ylim3d(0, finishedCut.shape[0])
ax.set_zlabel('Z')
ax.set_zlim3d(0, 1)

plt.show()

print("finish")