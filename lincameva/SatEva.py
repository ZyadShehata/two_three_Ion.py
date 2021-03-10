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

from Classes.EventPreProcessor import EventPreProcessor
from Classes.EinsteinCorrection import EinsteinCorrection

Detuning = 30e6

#Antibunching fitting function
def anti_bunching_fit(x, a, b, c, d, e):
    #a = 0.78
    #b = 129.87e6
    #c = 235e6
    #d = b/c
    #d = 0
    #return 1-a*np.exp(-b*x)*(np.cos(c*x)+(b/c)*np.sin(c*x))
    x = np.abs(x-d)
    return e*1.0-a*np.exp(-b*(x))*(np.cos(c*(x)))

def linear_fit(x, a, b):
    return a*x+b

def doSatFile(path, bins=360):

    global Detuning

    lastbin = int(0.9638888 * float(bins))

    binDurationMaster = 23.616e-12
    binDurationSlave = 22.22e-12
    totalBinDuration = 3700./float(bins)
    totalDurMaster = totalBinDuration * binDurationMaster
    totalDurSlave = totalBinDuration * binDurationSlave

    files = glob(path[0] + "/*.photons")

    addedMaster = [0]
    addedSlave = [0]

    for file in files:
        data = Vault(file)
        h = histogram(data.data.master.photons.tac[:], 1, 3701, bins)
        hTwo = histogram(data.data.slave.photons.tac[:], 1, 3701, bins)
        plt.clf()


        if (len(addedMaster) == 1):
            addedMaster = h
            addedSlave = hTwo
        else:
            addedMaster = np.add(addedMaster, h)
            addedSlave = np.add(addedSlave, hTwo)

    baselineMaster = np.mean(addedMaster[lastbin - 10:lastbin])
    baselineSlave = np.mean(addedSlave[lastbin - 10:lastbin])
    normedMaster = addedMaster / baselineMaster
    normedSlave = addedSlave / baselineSlave

    normedAndClippedMaster = normedMaster[:lastbin]
    normedAndClippedSlave = normedSlave[:lastbin]

    xvalsMaster = np.array(range(len(normedAndClippedMaster))) - np.argmin(normedAndClippedMaster)
    xvalsSlave = np.array(range(len(normedAndClippedSlave))) - np.argmin(normedAndClippedSlave)

    minMasterIndex = np.argmin(normedAndClippedMaster)
    minSlaveIndex = np.argmin(normedAndClippedSlave)

    timedXvalsMaster = xvalsMaster * totalDurMaster
    timedXvalsSlave = xvalsSlave * totalDurSlave


    poptM, pcovM = curve_fit(anti_bunching_fit, timedXvalsMaster, normedAndClippedMaster, [0.88, 0.5e8, 15e7, 0., 1.0])
    poptS, pcovS = curve_fit(anti_bunching_fit, timedXvalsSlave, normedAndClippedSlave, [0.88, 0.5e8, 15e7, 0., 1.0])
    #popt = [0.88, 0.5e8, 15e7, 0.]
    print("timing diffs for " + str(path[1]) + " uW")
    print("coarse diffs:")
    print("Master: " + str(minMasterIndex * totalDurMaster))
    print("Slave: " + str(minMasterIndex * totalDurSlave))
    print("fine diffs:")
    print("Master: " + str(poptM[3]))
    print("Slave: " + str(poptS[3]))
    print("Total diffs:")
    print("Master: " + str(minMasterIndex * totalDurMaster + poptM[3]))
    print("Slave: " + str(minMasterIndex * totalDurSlave + poptS[3]))
    print()

    plt.plot(timedXvalsMaster, normedAndClippedMaster, label="Master")
    plt.plot(timedXvalsSlave, normedAndClippedSlave, label="Slave")
    #saturation_param = 1.0
    saturation_paramM = (((1 / (2 * np.pi)) * poptM[2] ** 2) / (2)) / ((Detuning) ** 2 + (((1 / (7.7e-9)) ** 2) / (4)))
    saturation_paramS = (((1 / (2 * np.pi)) * poptS[2] ** 2) / (2)) / ((Detuning) ** 2 + (((1 / (7.7e-9)) ** 2) / (4)))
    plt.text(1e-8, 0.35, r'Master: $s \approx$' + str(round(saturation_paramM, 2)) + r' Slave: $s \approx$' + str(round(saturation_paramS, 2)),
             bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5})

    plt.plot(np.linspace(timedXvalsMaster[0], timedXvalsMaster[-1], num=1000), anti_bunching_fit(np.linspace(timedXvalsMaster[0], timedXvalsMaster[-1], num=1000), *poptM), label="Fit Master")
    plt.plot(np.linspace(timedXvalsSlave[0], timedXvalsSlave[-1], num=1000), anti_bunching_fit(np.linspace(timedXvalsSlave[0], timedXvalsSlave[-1], num=1000), *poptS), label="FitSlave")
    plt.title(str(path[1]) + r"$\mu W$")
    plt.ylabel(r"$g^{(2)}$ Signal")
    plt.xlabel(r"time in [s]")
    plt.legend(loc="best")
    plt.savefig(str(path[1]) + ".pdf", transparent=True)
    plt.savefig(str(path[1]) + ".png", transparent=True)
    #plt.show()

    return (saturation_paramM + saturation_paramS)/2.

#read raw data
#data = Vault(r'C:\Users\Photonscore\Desktop\Testing\Saturation\OneIon25uW.photons')

"""
imagMaster = histogram(data.data.master.photons.x[:], 0, 4000, 400, data.data.master.photons.y[:])
plt.imshow(imagMaster)
plt.title("Master Image")
#plt.show()

imagSlave = histogram(data.data.slave.photons.x[:], 0, 4000, 400, data.data.slave.photons.y[:])
plt.imshow(imagSlave)
plt.title("Slave Image")
#plt.show()
"""
#h = histogram(data.data.master.photons.tac[:], 1, 3701, 360)
#hTwo = histogram(data.data.slave.photons.tac[:], 1, 3701, 360)
"""
plt.plot(range(hTwo.size), h)
plt.plot(range(hTwo.size), hTwo)
plt.title("Tac Separate Histo")
plt.show()
"""


satFiles = [
    #[r'E:\RawData\SpatialIonsMainz\Measurement\01_08_2018_Saturation\10uW', 10],
    [r'E:\RawData\SpatialIonsMainz\Measurement\01_08_2018_Saturation\10uW2', 10],
    [r'E:\RawData\SpatialIonsMainz\Measurement\01_08_2018_Saturation\13uW', 13],
    #[r'E:\RawData\SpatialIonsMainz\Measurement\01_08_2018_Saturation\15uW', 15],
    #[r'E:\RawData\SpatialIonsMainz\Measurement\01_08_2018_Saturation\17_5uW', 17.5],
    #[r'E:\RawData\SpatialIonsMainz\Measurement\01_08_2018_Saturation\20uW', 20],
    [r'E:\RawData\SpatialIonsMainz\Measurement\01_08_2018_Saturation\20uWCorrectDet', 20],
    [r'E:\RawData\SpatialIonsMainz\Measurement\01_08_2018_Saturation\25uW', 25],
    [r'E:\RawData\SpatialIonsMainz\Measurement\01_08_2018_Saturation\25uW2', 25]
]

powers = []
saturations = []


for item in satFiles:
    powers.append(item[1])
    saturations.append(doSatFile(item, 160))

powers = np.array(powers)
saturations = np.array(saturations)

popt, pcov = curve_fit(linear_fit, powers, saturations)

plt.plot(powers, saturations, 'o')
plt.plot(powers, linear_fit(powers, *popt))
plt.text(14, 0.65, r'Saturation param: $s \approx$' + str(round(popt[0], 6)) + r" x P + " + str(round(popt[1], 3)),
             bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5})
plt.xlabel(r"power in $\mu W$")
plt.ylabel(r'saturation Param')
plt.title(r'Saturation vs. Power')
plt.show()

#plt.plot(np.array(range(-150, 150)), anti_bunching_fit(np.array(range(-150, 150)), 0.6, 0.02, 0.1, 0.5))
#plt.show()

print("finish")