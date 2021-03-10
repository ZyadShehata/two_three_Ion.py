import numpy as np
import matplotlib.pyplot as plt

histo = np.zeros((2,72,72,4096), dtype=np.int32)

#definition of histo functions
def antiBunch(x,a,b,c,d,e):
    x = x -2048
    x = np.abs(x - d)
    return np.array(np.around(e * 1.0 - a * np.exp(-b * x) * (np.cos(c * x))), dtype=np.int32)

def bunch(x,a,b,c):
    x = x - 2048
    x = np.abs(x)
    return np.array(np.around(a * np.exp((-1) * (1 / b) * x) + c), dtype=np.int32)

def createBaseLineNoise(binheight):
    return np.array(np.around((np.random.rand(4096) - 0.5)*2*np.sqrt(binheight)), dtype=np.int32)

def getSameChannels(histo, detector = 0):
    len = histo.shape[1]
    out = np.zeros((len, histo.shape[3]), dtype=np.int32)
    for i in range(len):
        out[i] = histo[detector][i][i]
    return out

def getCrossChannels(histo, startChannel, detector = 0):
    len = histo.shape[1]
    out = np.zeros((len, histo.shape[3]), dtype=np.int32)
    for i in range(len):
        out[i] = histo[detector][startChannel][i]
    return out

#simul params
baseline = 150000
binner = 64

xes = np.array(range(4096))
#create testing histo

#params exp-decay (5 cases)
paramsSemiBunch = [0.233*baseline, 600, baseline]
paramsFullBunch = [0.5*baseline, 600, baseline]
paramsLaserLike = [0.001*baseline, 2000, baseline]

#params antibunch (3 cases)
paramsSemiAnti = [0.233*baseline, 1./850., (2*np.pi/1550), 0.0, baseline]
paramsFullAnti = [0.55*baseline, 1./850., (2*np.pi/1550), 0.0, baseline]

noise = baseline*binner

for i in range(72):
    selector = i % 8
    vals = np.array([])
    if (selector == 0 or selector == 4):
        #laserlike
        vals = bunch(xes, *paramsLaserLike) + createBaseLineNoise(noise)
    if (selector == 1 or selector == 3):
        #semibunch
        vals = bunch(xes, *paramsSemiBunch) + createBaseLineNoise(noise)
    if (selector == 2):
        #bunch
        vals = bunch(xes, *paramsFullBunch) + createBaseLineNoise(noise)
    if (selector == 5 or selector == 7):
        #semi anti bunch
        vals = antiBunch(xes, *paramsSemiAnti) + createBaseLineNoise(noise)
    if (selector == 6):
        #full anti bunch
        vals = antiBunch(xes, *paramsFullAnti) + createBaseLineNoise(noise)

    histo[0][i][i] = vals

#test plotting
rawVals = getSameChannels(histo)
#rebin second axis
#how many rebins?
rebinner = binner#factorial of 2 (4, 8, 16, 32, ...)!
rebinned = np.zeros((rawVals.shape[0], int(rawVals.shape[1]/rebinner)), dtype=np.float64)
for i in range(rawVals.shape[0]):
    reshapedLine = np.sum(rawVals[i].reshape(-1, rebinner), axis=1)
    valOne = np.mean(reshapedLine[0:5])
    valTwo = np.mean(reshapedLine[57:63])
    totaloMean = max(np.mean([valOne, valTwo]), 1)
    rebinned[i] = 1./totaloMean * np.sum(rawVals[i].reshape(-1, rebinner), axis=1)
rebinned = rebinned.T

plt.imshow(rebinned, cmap='hot')
plt.colorbar()
plt.title("imag")
plt.show()

#zerovalPlot
bins =  rebinned.shape[0]
stop = int(bins/2)
start = int(stop-1)
zeroedOne = rebinned[:][start]
zeroedtwo = rebinned[:][stop]
zeroedTotal = np.add(zeroedOne, zeroedtwo)

plt.plot(zeroedTotal)
plt.show()


print("finish")