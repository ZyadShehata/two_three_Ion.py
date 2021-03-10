import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import interpolation

from photonscore.python.flim import read
from photonscore.python.histogram import histogram
from photonscore.python.vault import Vault

from matplotlib.colors import colorConverter
import matplotlib as mpl

from glob import glob

#from bokeh.models import HoverTool
#from bokeh.io import output_notebook

from Classes.EventPreProcessor import EventPreProcessor
from Classes.EinsteinCorrection import EinsteinCorrection


def oneDFringes(x, Amplitude, Phase, Period, Offset):
    #return 10000*np.sin(Period*x)+Offset
    return Amplitude*np.sin(Period*x+Phase)+Offset


data = Vault(r"E:\RawData\SpatialIonsMainz\Measurement\01_09_2018_FirstMeas\17uW53.photons")

masterX = data.data.master.photons.x[:].astype(np.int64) - 2048
masterY = data.data.master.photons.y[:].astype(np.int64) - 2048
masterCoords = np.array([
    masterX,
    masterY
]).T

rotAngle = (-29.72883115886924/180.)*np.pi
rotMatrix = np.array([
    [np.cos(rotAngle), -1*np.sin(rotAngle)],
    [np.sin(rotAngle), np.cos(rotAngle)]
])

rotatedMasterCoords = rotMatrix.dot(masterCoords.T).T + 2048

#imagMaster = histogram(data.data.master.photons.x[:], 0, 4000, 400, data.data.master.photons.y[:])
#plt.imshow(imagMaster)
#plt.title("Master Image Unrotated")
#plt.show()

imagMaster = histogram(rotatedMasterCoords.T[0], 0, 4000, 400, rotatedMasterCoords.T[1])
#plt.imshow(imagMaster)
#plt.title("Master Image Rotated")
#plt.show()

#print("master finished")

slaveX = data.data.slave.photons.x[:].astype(np.int64) - 2048
slaveY = data.data.slave.photons.y[:].astype(np.int64) - 2048
slaveCoords = np.array([
    slaveX,
    slaveY
]).T

rotAngle = (-5.19673564264405/180.)*np.pi
rotMatrix = np.array([
    [np.cos(rotAngle), -1*np.sin(rotAngle)],
    [np.sin(rotAngle), np.cos(rotAngle)]
])
#rotate
rotatedSlaveCoords = rotMatrix.dot(slaveCoords.T).T

copiedRotatedSlaveCoords = rotatedSlaveCoords

#translate
translateX = -65
translateY = -50

#mirror x-axis
rotatedSlaveCoords = np.array([
    (np.array(rotatedSlaveCoords.T[0]) * -1) + translateX,
    np.array(rotatedSlaveCoords.T[1] + translateY),
]).T
#reOffset
rotatedSlaveCoords = rotatedSlaveCoords + 2048

#imagSlave = histogram(data.data.slave.photons.x[:], 0, 4000, 400, data.data.slave.photons.y[:])
#plt.imshow(imagSlave)
#plt.title("Slave Image Unrotated")
#plt.show()

imagSlave = histogram(rotatedSlaveCoords.T[0], 0, 4000, 400, rotatedSlaveCoords.T[1])
#plt.imshow(imagSlave)
#plt.title("Slave Image Rotated")
#plt.show()


#optimzation process

Steps = range(-200,0, 2)
visibilities = []
for step in Steps:
    break
    print(str(step))
    temporalCopy = copiedRotatedSlaveCoords
    # translate
    translateX = step
    translateY = -50

    # mirror x-axis
    temporalCopy = np.array([
        (np.array(temporalCopy.T[0]) * -1) + translateX,
        np.array(temporalCopy.T[1] + translateY),
    ]).T
    # reOffset
    temporalCopy = temporalCopy + 2048

    # imagSlave = histogram(data.data.slave.photons.x[:], 0, 4000, 400, data.data.slave.photons.y[:])
    # plt.imshow(imagSlave)
    # plt.title("Slave Image Unrotated")
    # plt.show()

    imagSlave = histogram(temporalCopy.T[0], 0, 4000, 400, temporalCopy.T[1])
    reducedData = np.sum(imagSlave[125:340, 150:340], axis=0)
    xdata = np.array(range(len(reducedData)))
    popt, pcov = curve_fit(oneDFringes, xdata, reducedData, p0=[4000, -5100, 0.15, 25200])
    #popt = [4000, -5100, 0.15, 25200]
    #plt.plot(xdata, reducedData)
    #plt.plot(xdata, oneDFringes(xdata, *popt))
    #plt.title("Rotation: " + str(angle))
    #plt.show()
    visibility = (popt[0])/(popt[3])
    visibilities.append(np.abs(visibility))
    #print("did " + str(angle) + " degrees")
#rough estimation
#visibilities = np.array(visibilities)
#plt.plot(Steps, visibilities)
#plt.show()
#minStep = Steps[visibilities.argmin()]

minStep = -69

#translate
translateX = minStep
translateY = -50

#mirror x-axis
rotatedSlaveCoords = np.array([
    (np.array(copiedRotatedSlaveCoords.T[0]) * -1) + translateX,
    np.array(copiedRotatedSlaveCoords.T[1] + translateY),
]).T
#reOffset
rotatedSlaveCoords = rotatedSlaveCoords + 2048

#imagSlave = histogram(data.data.slave.photons.x[:], 0, 4000, 400, data.data.slave.photons.y[:])
#plt.imshow(imagSlave)
#plt.title("Slave Image Unrotated")
#plt.show()

imagSlave = histogram(rotatedSlaveCoords.T[0], 0, 4000, 400, rotatedSlaveCoords.T[1])



#opaque stuff
# generate the colors for your colormap
color1 = colorConverter.to_rgba('white')
color2 = colorConverter.to_rgba('black')

# make the colormaps
cmap1 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap',['green','blue'],256)
cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2',[color1,color2],256)

cmap2._init() # create the _lut array, with rgba values

# create your alpha array and fill the colormap with them.
# here it is progressive, but you can create whathever you want
alphas = np.linspace(0, 0.5, cmap2.N+3)
cmap2._lut[:,-1] = alphas

f = plt.figure()
f.add_subplot(2,2, 1)
plt.title("imag Master")
plt.imshow(imagMaster)
f.add_subplot(2,2, 2)
plt.title("imag Slave")
plt.imshow(imagSlave)
f.add_subplot(2,2,3)
plt.title("Overlay")
plt.imshow(imagMaster, interpolation='nearest', cmap=cmap1, origin='lower')
plt.imshow(imagSlave, interpolation='nearest', cmap=cmap2, origin='lower')
f.add_subplot(2,2,4)
plt.title("subtract")
plt.imshow(imagSlave/np.amax(imagSlave) - imagMaster/np.amax(imagMaster))
plt.show(block=True)

print("finish")
