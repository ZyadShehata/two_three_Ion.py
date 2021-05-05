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

    return mergedImag, errorsRelative
    return finishedNormalied, errorsRelative

def spatialoFunc(x, x0, spatMod, offsetx, offsety, sat, dwf, loss=1):
    return (((sat**2)*np.cos((spatMod*(x-offsetx)-x0)/2)**2*dwf*loss)/((sat+np.cos(x0)*dwf)*(sat+np.cos(spatMod*(x-offsetx))*dwf))) + offsety


sns.set()
sns.set_style("white")
sns.set_style("ticks")

#Setup variables
bins= 144
activeDiameter = 0.0254*0.98
sizePerBin = activeDiameter/bins
#sizePerBin = 220e-6
startOne = 10
endOne = 80
lensRadius = 0.034
FocalDistance = 0.6
distanceDetector = (FocalDistance/lensRadius)*(activeDiameter)
#distanceDetector = 0.4482
magnification = 15.
lensFNumber = 1.6
numericalAperture = 1/(2*lensFNumber)
abbeResolution = 397e-9/(2*numericalAperture)

histos = np.load("../data/AllDAta.bin.npy")
gOne = np.load("../data/alldataG1.npy")
gOneTot = np.load("../data/GOneTot.npy")

imagBinnedMaster, imagBinnedSlave = gOne
totaloBinned = np.add(imagBinnedSlave[1], imagBinnedMaster[1])
totaloBinned = totaloBinned / np.sum(totaloBinned)



dummyData, ErrorsRelative = getTimeSlice(histos, 47, gOne)
finishedCut = dummyData[:,startOne:endOne]

xs = np.array(range(finishedCut.shape[1]), dtype=np.float)
#translate to real world coordinates
realXes = (xs - len(xs)/2)*sizePerBin
verts = []
offsetbase = 41
zslook = np.array([41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55], dtype=np.float)
#zslook = np.flip(zslook)
zs = range(len(zslook))
minimum = 0
maximum = 0

spatialFrequencies = []
spatialErrors = []
fitparams = []
covmatrices = []
phases = []
offsets = []


fig, axs = plt.subplots(3, sharex=True, sharey=True)
for z in zs:
    temporalStart, ErrorsRelative = getTimeSlice(histos, int(zslook[int(z)]), gOne)
    ErrorsRelative = ErrorsRelative[startOne:endOne]
    meanval = np.mean(temporalStart[2:6])

    temporalCut = temporalStart[:,startOne:endOne]
    temporalCut = np.divide(temporalCut, np.amax(meanval))

    ys = temporalCut[int(temporalCut.shape[0]/2)-1]


    #calculate phase offset
    offset = zslook[z] - offsetbase
    realOffset = offset * sizePerBin
    offsettedXes = realXes+realOffset
    offsettedXes = offsettedXes / distanceDetector
    offsettedMin = np.amin(offsettedXes)
    offsettedMax = np.amax(offsettedXes)
    # check global min an max
    maximum = np.amax([offsettedMax, maximum])
    minimum = np.amin([offsettedMin, minimum])


    # fit data to model
    popt, pcov = curve_fit(spatialoFunc, offsettedXes[1:-1], ys[1:-1],
                           p0=[1.4 * np.pi, 1590, 0, 0.24, 0.9, 0.36, 0.8])

    spatialFrequencies.append(popt[1])
    spatialErrors.append(np.sqrt(pcov[1][1]))
    fitparams.append(popt)
    covmatrices.append(pcov)
    phases.append(popt[0])
    offsets.append(popt[2])

    #plot it
    #plt.clf()
    #plt.title(str(z))
    #plt.plot(offsettedXes, ys, "o")
    #plt.plot(np.linspace(offsettedXes[0], offsettedXes[-1], num=200), spatialoFunc(np.linspace(offsettedXes[0], offsettedXes[-1], num=200), *popt))
    #plt.ylim((0,1.6))
    #plt.show()

    if (z == 0):
        #axs[0].set_title("Position: " + str(z))
        (_, caps, _) = axs[0].errorbar(offsettedXes, ys, yerr=np.multiply(ys, ErrorsRelative), label="data", fmt="o", markersize=3, capsize=1.5)
        for cap in caps:
            cap.set_markeredgewidth(1)
        axs[0].plot(np.linspace(offsettedXes[0], offsettedXes[-1], num=200), spatialoFunc(np.linspace(offsettedXes[0], offsettedXes[-1], num=200), *popt))
    if (z == 10):
        #axs[1].set_title("Position: " + str(z))
        (_, caps, _)= axs[1].errorbar(offsettedXes, ys, yerr=np.multiply(ys, ErrorsRelative), label="data", fmt="o", markersize=3, capsize=1.5)
        for cap in caps:
            cap.set_markeredgewidth(1)
        axs[1].plot(np.linspace(offsettedXes[0], offsettedXes[-1], num=200),
                    spatialoFunc(np.linspace(offsettedXes[0], offsettedXes[-1], num=200), *popt))
    if (z == 14):
        #axs[2].set_title("Position: " + str(z))
        (_, caps, _) = axs[2].errorbar(offsettedXes, ys, yerr=np.multiply(ys, ErrorsRelative), label="data", fmt="o", markersize=3, capsize=1.5)
        for cap in caps:
            cap.set_markeredgewidth(1)
        axs[2].plot(np.linspace(offsettedXes[0], offsettedXes[-1], num=200),
                    spatialoFunc(np.linspace(offsettedXes[0], offsettedXes[-1], num=200), *popt))

    #fit theory, spit out virtual ion distance for each set
    ys[0], ys[-1] = 0, 0
    verts.append(list(zip(offsettedXes*1000, ys)))


#proc total plots
fig.savefig("./Spatial/selectedpositions.pdf")
#process spaciel frequencies and errors
meanSpatFreq = np.mean(spatialFrequencies)
propagatedError = (1/len(spatialErrors)*np.sqrt(np.sum(np.array(spatialErrors)**2)))
relativeError = propagatedError/meanSpatFreq
#calculate ion distance
magnifiedDistance = (meanSpatFreq/(2*np.pi))*397e-9
realDistance = magnifiedDistance/magnification

print("spatial Frequency: " + str(meanSpatFreq) + "+-" + str(propagatedError) + ": +-" + str(np.round(propagatedError*100/meanSpatFreq, 2)) + "%")
print("ion distance: " + str(realDistance) + "+-" + str(realDistance*relativeError) + "m")
print("Abbe Resolution: " + str(abbeResolution))
print("Improvement over Abbe: " + str(np.round(abbeResolution/(realDistance*relativeError),2)))

#plt.show()
plt.clf()
plt.tight_layout()
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.view_init(elev=45., azim=280.)
facecolors = [cm.jet(x) for x in np.linspace(0., 1., num=len(zs))]
black = (0,0,0,0.7)
poly = PolyCollection(verts, facecolors=facecolors, edgecolors=(black,))
poly.set_alpha(0.7)
ax.add_collection3d(poly, zs=zs, zdir='y')

ax.set_xlabel("\n" + r'Observation angle $[mrad]$', linespacing=2.)
ax.set_xlim3d(minimum*1000, maximum*1000)
ax.set_ylabel('\nStart Detector', linespacing=2.)
ax.set_ylim3d(0, len(zs))
ax.set_zlabel("\n" + r'$g^{(2)}\left( \vec{x}_1,\vec{x}_2 \right)$', linespacing=2.)
ax.set_zlim3d(0, 1.2)

#plt.show()
plt.savefig("./Spatial/adjustedPhase.pdf")

print("finish")