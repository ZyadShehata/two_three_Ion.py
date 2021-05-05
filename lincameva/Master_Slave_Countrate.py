from glob import glob
from photonscore.python.vault import Vault
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime

#functions go here

def getEdges(data, lowerEdgeThreshold = -2000, upperEdgeThreshold = 2000):
    signedData = np.array(data, dtype=np.int64)
    countRates = np.diff(signedData[::1000])
    edgesraw = np.diff(countRates)
    edgesfalling = edgesraw < lowerEdgeThreshold
    edgesrising = edgesraw > upperEdgeThreshold
    return edgesrising, edgesfalling

def cleanEdges(edgeIds):
    stepsize = 3
    return np.split(edgeIds, np.where(np.diff(edgeIds) > stepsize)[0] + 1)

def getCleanedRising(cleanedArray):
    output = []
    try:
        for singleEdgeBlock in cleanedArray:
            output.append(singleEdgeBlock[-1])
        return np.array(output)
    except IndexError:
        return np.array([])

def getCleanedFalling(cleanedArray):
    output = []
    try:
        for singleEdgeBlock in cleanedArray:
            output.append(singleEdgeBlock[0])
        return np.array(output)
    except IndexError:
        return np.array([])

def getLengthOfRegionsOfInterest(data):
    risingEdges, fallingEdges = getEdges(data)
    #do translation of boolean arrays to indexes
    risingIds = np.where(risingEdges == True)
    fallingIds = np.where(fallingEdges == True)
    #clean indexes
    risingCleaned = getCleanedRising(cleanEdges(risingIds[0]))
    fallingCleaned = getCleanedFalling(cleanEdges(fallingIds[0]))

    #case 1: no edges at all
    if (len(risingCleaned) == 0 and len(fallingCleaned) == 0):
        #check if countrate is above 1500
        if (np.mean(np.diff(data[::1000])) > 4000):
            return float(len(data))/1000.

    lenOfGoodTime = 0

    risingFirst = False
    fallingFirst = False
    #case 2: rising edge first
    if (len(risingCleaned) != 0):
        if (len(fallingCleaned) == 0):
            risingFirst = True
        else:
            if (risingCleaned[0] < fallingCleaned[0]):
                risingFirst = True

    #case 3: falling edge first
    if (len(fallingCleaned) != 0):
        if (risingFirst == False):
            fallingFirst = True

    #Runner for rising First
    if (risingFirst):
        mainArray = risingCleaned
        for i in range(len(mainArray)):
            rising = mainArray[i]
            #check if there is falling[i]
            if (i > len(fallingCleaned) - 1):
                #no corresponding falling edge, calculate time till end fo file
                timeDifference = float(len(data))/1000. - rising
                # check if selected time has single ion case or fuck up!
                meanRateOfBlock = (data[-1] - data[rising * 1000]) / timeDifference
                if (meanRateOfBlock > 4000):
                    lenOfGoodTime += timeDifference
                break
            falling = fallingCleaned[i]
            timeDifference = falling - rising
            #check if selected time has single ion case or fuck up!
            meanRateOfBlock = (data[falling*1000] - data[rising*1000])/timeDifference
            if (meanRateOfBlock > 4000):
                lenOfGoodTime += timeDifference

    #Runner for falling frst
    if (fallingFirst):
        pass

    return lenOfGoodTime

    #runtime goes here
if __name__ == "__main__":
    allFiles = glob("d:\\Zusammen\\*.photons")
    lenOfFiles = len(allFiles)
    masterCounts = np.zeros(len(allFiles), dtype=np.float)
    slaveCounts = np.zeros(len(allFiles), dtype=np.float)

    totalSecondsOfExperiment = 0
    totalSecondsOfExperimentOfMasterGood = 0
    totalSecondsOfExperimentOfSlaveGood = 0

    #only take one file as bad example
    #idOfBadGuy = 19
    #tempArray = [allFiles[idOfBadGuy]]
    #allFiles = tempArray

    i = 0
    for singleFile in allFiles:
        data = Vault(singleFile)
        photons = data.data.master.photons.a.shape[0]
        # Get total measurement time
        duration = data.data.master.status.daq_counts.shape[0]/1000
        meanCount = photons/duration
        masterCounts[i] = meanCount
        photons = data.data.slave.photons.a.shape[0]
        duration = data.data.slave.status.daq_counts.shape[0] / 1000
        meanCount = photons / duration
        slaveCounts [i] = meanCount
        totalSecondsOfExperiment += duration

        #Evaluate proper count rates
        ratesMaster = np.diff(data.data.master.status.daq_counts[::1000])
        ratesSlave = np.diff(data.data.slave.status.daq_counts[::1000])

        edgesMaster = getLengthOfRegionsOfInterest(data.data.master.status.daq_counts[:])
        edgesSlave = getLengthOfRegionsOfInterest(data.data.slave.status.daq_counts[:])

        totalSecondsOfExperimentOfMasterGood += edgesMaster
        totalSecondsOfExperimentOfSlaveGood += edgesSlave

        #plt.clf()
        #plt.figure()
        #plt.plot(ratesMaster)
        #plt.plot(ratesSlave)
        #plt.show()

        print(str(i) + "/" + str(lenOfFiles) + "Master good: " + str(edgesMaster) + " Slave good: " + str(edgesSlave))
        i += 1

    a = datetime.timedelta(seconds=totalSecondsOfExperiment)
    print("total measurement time was: " + str(totalSecondsOfExperiment) + " that is: " + str(a))
    print("total master good time was: " + str(totalSecondsOfExperimentOfMasterGood) + " that is: " + str(a))
    print("total slave good time was: " + str(totalSecondsOfExperimentOfSlaveGood) + " that is: " + str(a))
    dutyCycleMaster = totalSecondsOfExperimentOfMasterGood / totalSecondsOfExperiment
    dutyCycleSlave = totalSecondsOfExperimentOfSlaveGood / totalSecondsOfExperiment
    print("duty cycle master: " + str(dutyCycleMaster))
    print("duty cycle slave: " + str(dutyCycleSlave))

    #plt.figure()
    #plt.hist(masterCounts, label = 'Master')
    #plt.hist(slaveCounts, label = 'Slave')
    #plt.title("Master Counts")
    #plt.xlabel('count rate (Hz)')
    #plt.ylabel('No. of files')
    #plt.xlim(3500,10045)
    #plt.legend()

    #plt.show()

    print("finish")