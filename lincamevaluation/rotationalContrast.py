import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib import colors as mcolors
from matplotlib import cm
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.ndimage import rotate

sns.set()
sns.set_style("white")
sns.set_style("ticks")

countsMax = 50.
countsMax = 1800.
gOneMax = 1.13e7
divider = countsMax/gOneMax

#WROOOOOONG
def angularModel(x, a, f, d, e, c):
    return (np.e**(e*np.abs(x+c))*a*(np.cos(f*(x+c))**2)+d)

def gaussian(x, mu, sig, amplitude, offset):
    return amplitude*np.exp(-np.power((x - mu)/sig, 2.)/2)+offset

def visibilityError(max, min, deltaMax, deltaMin):
    return np.sqrt((((2*max)/(max+min)**2)*deltaMax)**2+(((-2*max)/(max+min)**2)*deltaMin)**2)

def doAngle(angle, data, divider):

    gOneAdded = rotate(data, angle, axes=(1, 0), reshape=False)

    gOneSelected = gOneAdded[125:245,60:330]
    gOneProjected = np.sum(gOneSelected, axis=0)

    #plt.title("Angle: " + str(angle) + " degrees")
    #plt.imshow(gOneAdded)
    #plt.show()
    #plt.clf()

    #plt.title("Angle: " + str(angle) + " degrees")
    #plt.imshow(gOneSelected)
    #plt.show()
    #plt.clf()

    #plt.title("Angle: " + str(angle) + " degrees")
    #plt.plot(gOneProjected*divider)
    #plt.show()
    #plt.clf()

    max = np.amax(gOneProjected)*divider
    min = np.amin(gOneProjected)*divider
    deltaMax = np.sqrt(max)
    deltaMin = np.sqrt(min)

    contrast = float(max-min)/float(max+min)
    error = visibilityError(max, min, deltaMax, deltaMin)

    return contrast, error

#histos = np.load("AllDAta.bin.npy")
#gOne = np.load("alldataG1.npy")
gOneTot = np.load("GOneTot.npy")

gOneAdded = np.add(gOneTot[0], gOneTot[1])
#gOneSelected = gOneAdded[125:245,60:330]
#gOneProjected = np.sum(gOneSelected, axis=0)

Angles = np.linspace(-7.5, 15, num=6)
Angles = np.append(Angles, 0.)
Angles = np.sort(Angles)
fineAngles = np.linspace(-9, 16, num=200)
Contrasts = []
Errors = []
for angle in Angles:
    Contrast, Error = doAngle(angle, gOneAdded, divider)
    Contrasts.append(Contrast)
    Errors.append(Error)

startVals = [1.9, 7.0, 0.31, 0.06]
popt, pcov = curve_fit(gaussian, Angles, Contrasts, p0=startVals)

plt.plot(fineAngles, gaussian(fineAngles, *popt), label="fit")
(_, caps, _) = plt.errorbar(Angles, Contrasts, yerr=Errors, label="data", fmt="o", markersize=6, capsize=8)
for cap in caps:
    cap.set_markeredgewidth(1)
#plt.show()
plt.savefig("./Spatial/AngularPlot.pdf")

print("angular stuff in rad: " + str(popt[0]*(np.pi/180.0)) + "+-" + str(np.sqrt(pcov[0][0])*(np.pi/180.0)))

print("finish")