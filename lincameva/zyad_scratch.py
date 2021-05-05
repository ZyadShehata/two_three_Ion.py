from glob import glob
from photonscore.python.vault import Vault
import numpy as np
import matplotlib.pyplot as plt

allFiles = glob("d:\\Zusammen\\*.photons")
lenOfFiles = len(allFiles)
masterCounts = np.zeros(len(allFiles), dtype=np.float)
slaveCounts = np.zeros(len(allFiles), dtype=np.float)

i = 0
for singleFile in allFiles:
    data = Vault(singleFile)
    photons = data.data.master.photons.a.shape[0]
    duration = data.data.master.status.daq_counts.shape[0]/1000
    meanCount = photons/duration
    masterCounts[i] = meanCount
    photons = data.data.slave.photons.a.shape[0]
    duration = data.data.slave.status.daq_counts.shape[0] / 1000
    meanCount = photons / duration
    slaveCounts [i] = meanCount
    print(str(i) + "/" + str(lenOfFiles))
    i += 1

plt.figure()
plt.hist(masterCounts)
plt.hist(slaveCounts)
plt.show()

print("finish")

countpersecond = np.diff(data[::1000])
edgesraw = np.abs(diff(countpersecond))
edgesfalling = edgesraw[edgesraw < -2000]
edgesrising = edgesraw[edgesraw > -2000]
