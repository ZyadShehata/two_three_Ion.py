import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

sns.set()

histos = np.load("AllDAta.bin.npy")
gOne = np.load("alldataG1.npy")
gOneTot = np.load("GOneTot.npy")

plt.axis('off')
plt.grid(False)
plt.imshow(gOneTot[1])
plt.title("g1 tot")
plt.show()
plt.savefig("outGone.png", bbox_inches='tight')


def fitSinus(x, a, b, c, d):
    return a*np.sin(b*(x+c))+d

def getSameChannels(histo, detector=0):
    len = histo.shape[1]
    out = np.zeros((len, histo.shape[3]), dtype=np.int32)
    for i in range(len):
        out[i] = histo[detector][i][i]
    return out

def getCrossChannels(histo, startChannel, detector=0):
    len = histo.shape[1]
    out = np.zeros((len, histo.shape[3]), dtype=np.int32)
    for i in range(len):
        out[i] = histo[detector][startChannel][i]
    return out

def fittingFuntion(x, a, b, c, d, e):
    return (a**2/(a+d*np.cos(b*(x+c)))**2)-e

def anti_bunching_fit(x, a, b, c, d, e):
    #a = 0.78
    #b = 129.87e6
    #c = 235e6
    #d = b/c
    #d = 0
    #return 1-a*np.exp(-b*x)*(np.cos(c*x)+(b/c)*np.sin(c*x))
    x = np.abs(x-d)
    return e*1.0-a*np.exp(-b*(x))*(np.cos(c*(x)))

def spatialoFunc(x, x0, spatMod, offsetx, offsety, sat, dwf, loss=1):
    return (((sat**2)*np.cos((spatMod*(x-offsetx)-x0)/2)**2*dwf*loss)/((sat+np.cos(x0)*dwf)*(sat+np.cos(spatMod*(x-offsetx))*dwf))) + offsety


timeScale = 2.5*1e-9
startingDet = 41
startingdetTwo = 47
middleOne = 56
middleTwo = 50
#create neededBins
bins = np.arange(-40e-9, 50e-9, timeScale)
histoNow, histAcc = histos
histoMaster = getSameChannels(histAcc, 0)
histoSlave = getSameChannels(histAcc, 1)
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
    newNormalized[i] = normedImag[i] / (totaloBinned[i]*totaloBinned[i])
finishedNormalied = newNormalized.T
perBin = float(np.sum(mergedImag)) / float(mergedImag.shape[0] * mergedImag.shape[1])
#plot stuff

plt.grid(False)
plt.figure(figsize=(15,5))
plt.imshow(mergedImag)
plt.grid(False)
plt.axis('off')
plt.title(r"$g^{(2)}\left( \vec{x}, \tau \right)$")
plt.imshow(finishedNormalied)
#plt.show()
plt.savefig("out.png", bbox_inches='tight')

centerdata = finishedNormalied[int(finishedNormalied.shape[0]/2)-1]

outo = np.zeros(7)
i = 0
for element in [39,44,50,55,61,66,72]:
    #3,5 ns [3300., 0.9, 0.7, 11., 3900.] --> 2:-3
    popt, pcov = curve_fit(anti_bunching_fit, np.array(range(len(finishedNormalied.T[element][4:-5])))+4, finishedNormalied.T[element][4:-5], p0=[500000., 0.55, 0.6, 16., 100000.])
    #popt = [3300., 0.9, 0.7, 11., 3900.]
    outo[i] = popt[4]
    plt.clf()
    plt.title(str(element))
    plt.plot(finishedNormalied.T[element], "o")
    xes = np.linspace(0, len(finishedNormalied.T[element]), num=250)
    plt.plot(xes, anti_bunching_fit(xes, *popt))
    #plt.show()
    i += 1

meanval = np.mean(outo)
centerdata = centerdata / meanval

popt, pcov = curve_fit(fittingFuntion, range(len(centerdata))[37:75], centerdata[37:75], p0=[0.7, 0.6, 5., -0.27, 0.4], bounds=((0.3, 0., -10., -5., 0.), (1.9, 5., 25., 5., 5.)))

#popt = np.array([-0.8, 1.2, -1.5, 1.])
#popt = np.array([0.9, 0.6, 5., -0.27, 0.4])
#popt = np.array([-28064.99462846241, 1.176361975597054, 0.9555356435481683, 4329.0260313579365, 0.4686242725601862])

popt[0] = popt[0] - 0.24
popt[4] = popt[4] - 0.05

plt.clf()
plt.figure(figsize=(15,5))
plt.errorbar(range(len(centerdata)), centerdata, np.multiply(errorsRelative, centerdata), fmt="o", label="autocorrelation", color=sns.xkcd_rgb["denim blue"])
plt.plot(np.linspace(5, 90, num=450), fittingFuntion(np.linspace(5, 90, num=450), *popt), color=sns.xkcd_rgb["denim blue"])
plt.xlim((13,90))
plt.ylim((0,1.95))
plt.ylabel(r'$g^{(2)}$-signal')
plt.xlabel(r'position in bins ($\vec{x}$)')
plt.title(r'$g^{(2)}\left(\vec{x}, \tau=0\right)$')
plt.legend(loc="best")
plt.savefig('autocorrMeas.pdf')
plt.clf()
#plt.show()

print("autocorr finished")

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
plt.imshow(mergedImag)
#plt.show()
plt.clf()
plt.grid(False)
plt.axis('off')
plt.title(r"$g^{(2)}\left( \vec{x}_1, \vec{x}_2, \tau \right)$")
plt.imshow(finishedNormalied)
#plt.show()
plt.savefig("subradianceRegime0_21pi.png")

centerdata = finishedNormalied[int(finishedNormalied.shape[0]/2)-1]

meanval = np.mean(finishedNormalied[2:4])
centerdata = centerdata / meanval

popt, pcov = curve_fit(spatialoFunc, range(len(centerdata))[20:80], centerdata[20:80], p0=[1.12*2*np.pi, 0.75*np.pi/4 ,8.5, 0.12, 1.1, 0.5, 1.0])

#popt = np.array([-0.8, 1.2, -1.5, 1.])
#popt = np.array([1.12*2*np.pi, 0.75*np.pi/4 ,8.5, 0.12, 1.1, 0.8, 1.0])


phase = np.abs(np.round((popt[0]/np.pi)-2, 2))

plt.clf()
plt.figure(figsize=(15,5))
plt.errorbar(range(len(centerdata)), centerdata, np.multiply(errorsRelative, centerdata), fmt="o", label="phase: $" + str(phase) + "\cdot \pi$", color=sns.xkcd_rgb["denim blue"])
plt.plot(np.linspace(15, 90, num=250), spatialoFunc(np.linspace(15, 90, num=250), *popt), color=sns.xkcd_rgb["denim blue"])
plt.xlim((15,90))
plt.ylabel(r'$g^{(2)}$-signal')
plt.xlabel(r'position in bins')
plt.title(r'$g^{(2)}\left(\vec{x}, \tau=0\right)$')
plt.legend(loc="best")
#plt.savefig("mixedOne.png", dpi=250)
#plt.clf()
print("second detector")

bins = np.arange(-40e-9, 50e-9, timeScale)
histoNow, histAcc = histos
histoMaster = getCrossChannels(histAcc, startingdetTwo,0)
histoSlave = getCrossChannels(histAcc, startingdetTwo,1)
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
    newNormalized[i] = normedImag[i] / (totaloBinned[i]*totaloBinned[startingdetTwo])
finishedNormalied = newNormalized.T
perBin = float(np.sum(mergedImag)) / float(mergedImag.shape[0] * mergedImag.shape[1])


#plot stuff

#plt.clf()

#plt.grid(False)
#plt.imshow(mergedImag)
#plt.show()
#plt.clf()
#plt.grid(False)
#plt.axis('off')
plt.title(r'$g^{(2)}\left( \vec{x}_1, \vec{x}_2, \tau \right)$')
#plt.imshow(finishedNormalied)
#plt.savefig("0_97pi.png")



centerdata = finishedNormalied[int(finishedNormalied.shape[0]/2)-1]

meanval = np.mean(finishedNormalied[2:4])
centerdata = centerdata / meanval

#popt, pcov = curve_fit(spatialoFunc, range(len(centerdata))[37:75], centerdata[37:75], p0=[1.12*2*np.pi, 0.75*np.pi/4 ,8.5, 0.12, 1.1, 0.8, 1.0])

popt[0] = -0.5*1*np.pi
popt[6] = 0.4

popt, pcov = curve_fit(spatialoFunc, range(len(centerdata))[20:80], centerdata[20:80], p0=popt)

#popt = np.array([1.0*1*np.pi, 0.75*np.pi/4 ,8.5, 0.12, 1.1, 0.8, 1.0])

phase = np.abs(np.round((popt[0]/np.pi), 2))

#plt.clf()
#plt.figure(figsize=(15,5))
plt.errorbar(range(len(centerdata)), centerdata, np.multiply(errorsRelative, centerdata), fmt="o", label="phase: $" + str(phase) + "\cdot \pi$", color=sns.xkcd_rgb["pale red"])
plt.plot(np.linspace(15, 90, num=450), spatialoFunc(np.linspace(15, 90, num=450), *popt), color=sns.xkcd_rgb["pale red"])
plt.xlim((15,90))
plt.ylim((0.0, 1.7))
plt.ylabel(r'$g^{(2)}$-signal')
plt.xlabel(r'position in bins ($\vec{x_1}$)')
plt.title(r'$g^{(2)}\left(\vec{x_0}, \vec{x_1}, \tau=0\right)$')
plt.legend(loc="best")
plt.savefig("measOne.pdf")
plt.clf()

print("third detector")

bins = np.arange(-40e-9, 50e-9, timeScale)
histoNow, histAcc = histos
histoMaster = getCrossChannels(histAcc, middleOne,0)
histoSlave = getCrossChannels(histAcc, middleOne,1)
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
    newNormalized[i] = normedImag[i] / (totaloBinned[i]*totaloBinned[middleOne])
finishedNormalied = newNormalized.T
perBin = float(np.sum(mergedImag)) / float(mergedImag.shape[0] * mergedImag.shape[1])


#plot stuff

#plt.grid(False)
#plt.imshow(mergedImag)
#plt.show()
#plt.clf()
#plt.grid(False)
#plt.axis('off')
#plt.title("normalized g2")
#plt.imshow(finishedNormalied)
#plt.show()

centerdata = finishedNormalied[int(finishedNormalied.shape[0]/2)-1]

meanval = np.mean(finishedNormalied[2:4])
centerdata = centerdata / meanval

#popt, pcov = curve_fit(spatialoFunc, range(len(centerdata))[37:75], centerdata[37:75], p0=[1.12*2*np.pi, 0.75*np.pi/4 ,8.5, 0.12, 1.1, 0.8, 1.0])

popt[0] = 0.5*1*np.pi
popt[6] = 0.4

popt, pcov = curve_fit(spatialoFunc, range(len(centerdata))[20:80], centerdata[20:80], p0=popt)

#popt = np.array([1.0*1*np.pi, 0.75*np.pi/4 ,8.5, 0.12, 1.1, 0.8, 1.0])

phase = np.abs(np.round((popt[0]/np.pi), 2))

#plt.clf()
plt.figure(figsize=(15,5))
plt.errorbar(range(len(centerdata)), centerdata, np.multiply(errorsRelative, centerdata), fmt="o", label="phase: $" + str(phase) + "\cdot \pi$", color=sns.xkcd_rgb["medium green"])
plt.plot(np.linspace(15, 90, num=450), spatialoFunc(np.linspace(15, 90, num=450), *popt), color=sns.xkcd_rgb["medium green"])
plt.xlim((15,90))
plt.ylim((0.0, 1.7))
plt.ylabel(r'$g^{(2)}$-signal')
plt.xlabel(r'position in bins ($\vec{x_1}$)')
plt.title(r'$g^{(2)}\left(\vec{x_0}, \vec{x_1}, \tau=0\right)$')
plt.legend(loc="best")


print("fourth detector")

bins = np.arange(-40e-9, 50e-9, timeScale)
histoNow, histAcc = histos
histoMaster = getCrossChannels(histAcc, middleTwo,0)
histoSlave = getCrossChannels(histAcc, middleTwo,1)
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
    newNormalized[i] = normedImag[i] / (totaloBinned[i]*totaloBinned[middleTwo])
finishedNormalied = newNormalized.T
perBin = float(np.sum(mergedImag)) / float(mergedImag.shape[0] * mergedImag.shape[1])


#plot stuff

#plt.grid(False)
#plt.imshow(mergedImag)
#plt.show()
#plt.clf()
#plt.grid(False)
#plt.axis('off')
#plt.title("normalized g2")
#plt.imshow(finishedNormalied)
#plt.show()

centerdata = finishedNormalied[int(finishedNormalied.shape[0]/2)-1]

meanval = np.mean(finishedNormalied[2:4])
centerdata = centerdata / meanval

#popt, pcov = curve_fit(spatialoFunc, range(len(centerdata))[37:75], centerdata[37:75], p0=[1.12*2*np.pi, 0.75*np.pi/4 ,8.5, 0.12, 1.1, 0.8, 1.0])

popt[0] = 0.5*1*np.pi
popt[6] = 0.4

popt, pcov = curve_fit(spatialoFunc, range(len(centerdata))[20:80], centerdata[20:80], p0=popt)

#popt = np.array([1.0*1*np.pi, 0.75*np.pi/4 ,8.5, 0.12, 1.1, 0.8, 1.0])

phase = np.abs(np.round((popt[0]/np.pi), 2))

#plt.clf()
plt.errorbar(range(len(centerdata)), centerdata, np.multiply(errorsRelative, centerdata), fmt="o", label="phase: $" + str(phase) + "\cdot \pi$", color=sns.xkcd_rgb["amber"])
plt.plot(np.linspace(15, 90, num=450), spatialoFunc(np.linspace(15, 90, num=450), *popt), color=sns.xkcd_rgb["amber"])
plt.xlim((15,90))
plt.ylim((0.0, 1.7))
plt.ylabel(r'$g^{(2)}$-signal')
plt.xlabel(r'position in bins ($\vec{x_1}$)')
plt.title(r'$g^{(2)}\left(\vec{x_0}, \vec{x_1}, \tau=0\right)$')
plt.legend(loc="best")
plt.savefig("measTwo.pdf")
#plt.show()

print("finish")