import numpy as np
from numba import jit, cuda, njit
import matplotlib.pyplot as plt

from photonscore.python.flim import read
from photonscore.python.histogram import histogram
from photonscore.python.vault import Vault


class CorrelatorLin:

    __histoNow = ""
    __histos = ""
    __imagAccumulatedM = ""
    __imagNowM = ""
    __imagAccumulatedS = ""
    __imagNowS = ""
    __imagAccumulatedBinnedM = ""
    __imagAccumulatedBinnedS = ""
    __filePath = ""
    __histoTimingLen = 40
    __spatialCorr = False
    __spatialBins = 144

    def __init__(self):
        self.__histos = np.zeros((2, self.__spatialBins, self.__spatialBins,4096), dtype=np.int32)
        self.__histosNow = np.zeros((2,self.__spatialBins, self.__spatialBins,4096), dtype=np.int32)
        self.__imagAccumulatedM = np.zeros((2,2))
        self.__imagAccumulatedS = np.zeros((2, 2))
        self.__imagNowM = np.zeros((2, 2))
        self.__imagNowS = np.zeros((2, 2))

        self.__imagAccumulatedBinnedM = np.array([range(self.__spatialBins), np.zeros(self.__spatialBins)])
        self.__imagAccumulatedBinnedS = np.array([range(self.__spatialBins), np.zeros(self.__spatialBins)])

    def loadPath(self, path):
        self.__filePath = path

    def resetHisto(self):
        self.__histos = np.zeros((2, self.__spatialBins, self.__spatialBins, 4096), dtype=np.int32)

    def getHistos(self):
        return self.__histoNow, self.__histos

    def getImages(self):
        return self.__imagNowM, self.__imagNowS, self.__imagAccumulatedM, self.__imagAccumulatedS

    def getBinnedG1(self):
        return self.__imagAccumulatedBinnedM, self.__imagAccumulatedBinnedS

    def loadAndAppendHisto(self, histo):
        self.__histos = np.add(self.__histos, histo)

    def setSpatialCorr(self, spatialBool):
        self.__spatialCorr = spatialBool

    def setSpatialBins(self, newBins):
        self.__spatialBins = newBins
        self.__init__()

    def correlate(self):
        #define rot and trans
        rotAngleM = (-29.72883115886924 / 180.) * np.pi
        rotMatrixM = np.array([
            [np.cos(rotAngleM), -1 * np.sin(rotAngleM)],
            [np.sin(rotAngleM), np.cos(rotAngleM)]
        ])
        rotAngleS = (-5.19673564264405 / 180.) * np.pi
        rotMatrixS = np.array([
            [np.cos(rotAngleS), -1 * np.sin(rotAngleS)],
            [np.sin(rotAngleS), np.cos(rotAngleS)]
        ])
        # translate
        translateX = -69
        translateY = -50
        #load File
        data = Vault(self.__filePath)
        #create Images
        masterX = data.data.master.photons.x[:].astype(np.int64) - 2048
        masterY = data.data.master.photons.y[:].astype(np.int64) - 2048
        masterCoords = np.array([
            masterX,
            masterY
        ]).T
        rotatedMasterCoords = rotMatrixM.dot(masterCoords.T).T + 2048
        imagMaster = histogram(rotatedMasterCoords.T[0], 0, 4000, 400, rotatedMasterCoords.T[1])

        slaveX = data.data.slave.photons.x[:].astype(np.int64) - 2048
        slaveY = data.data.slave.photons.y[:].astype(np.int64) - 2048
        slaveCoords = np.array([
            slaveX,
            slaveY
        ]).T

        # rotate
        rotatedSlaveCoords = rotMatrixS.dot(slaveCoords.T).T

        copiedRotatedSlaveCoords = rotatedSlaveCoords


        # mirror x-axis
        rotatedSlaveCoords = np.array([
            (np.array(rotatedSlaveCoords.T[0]) * -1) + translateX,
            np.array(rotatedSlaveCoords.T[1] + translateY),
        ]).T
        # reOffset
        rotatedSlaveCoords = rotatedSlaveCoords + 2048

        imagSlave = histogram(rotatedSlaveCoords.T[0], 0, 4000, 400, rotatedSlaveCoords.T[1])

        if (self.__imagAccumulatedM.shape[0] == 2):
            #init everything
            self.__imagNowM = imagMaster
            self.__imagNowS = imagSlave
            self.__imagAccumulatedM = imagMaster
            self.__imagAccumulatedS = imagSlave
        else:
            #images already ceated
            self.__imagNowM = imagMaster
            self.__imagNowS = imagSlave
            self.__imagAccumulatedM = np.add(imagMaster, self.__imagAccumulatedM)
            self.__imagAccumulatedS = np.add(imagSlave, self.__imagAccumulatedS)

        #define slicing func
        @jit(nopython=True)
        def getSliceIndex(input, bins):
            neededBinSize = int(3096/bins)
            index = (input - 450) // neededBinSize
            index = max(0, index)
            index = min(bins - 1, index)
            return index

        #do exact g1 with rights bins for normalization
        @jit(nopython=True)
        def createSlicedG1(xes, bins):
            imag = np.zeros(bins)
            neededBinSize = int(3096/bins)
            for x in xes:
                index = (x - 450) // neededBinSize
                index = max(0, index)
                index = min(bins - 1 , index)
                imag[int(index)] += 1
            return imag

        self.__imagAccumulatedBinnedM[1] = np.add(createSlicedG1(rotatedMasterCoords.T[0], self.__spatialBins), self.__imagAccumulatedBinnedM[1])
        self.__imagAccumulatedBinnedS[1] = np.add(createSlicedG1(rotatedSlaveCoords.T[0], self.__spatialBins), self.__imagAccumulatedBinnedS[1])

        #do correlation

        indexTacNonZero = np.nonzero(data.data.master.photons.tac[:])
        xesM = data.data.master.photons.x[:][indexTacNonZero]
        yesM = data.data.master.photons.y[:][indexTacNonZero]
        aatM = data.data.master.photons.aat[:][indexTacNonZero]
        tacM = data.data.master.photons.tac[:][indexTacNonZero]
        chanM = np.zeros(len(xesM))
        masterVals = np.array([
            xesM,
            yesM,
            aatM,
            tacM,
            chanM
        ]).T
        indexTacNonZero = np.nonzero(data.data.slave.photons.tac[:])
        xesS = data.data.slave.photons.x[:][indexTacNonZero]
        yesS = data.data.slave.photons.y[:][indexTacNonZero]
        aatS = data.data.slave.photons.aat[:][indexTacNonZero]
        tacS = data.data.slave.photons.tac[:][indexTacNonZero]
        chanS = np.ones(len(xesS))
        slaveVals = np.array([
            xesS,
            yesS,
            aatS,
            tacS,
            chanS
        ]).T

        totalVals = np.concatenate((masterVals, slaveVals))
        totalVals = totalVals[np.argsort(totalVals[:, 2])]
        diffs = np.diff(totalVals.T[4])
        id_pos = np.where(diffs == 1)[0]
        id_pos_sec = id_pos + 1
        newValsX_pos = []
        newValsY_pos = []
        index_pos = []
        index_sec_pos = []

        for i in id_pos:
            sum = totalVals[i][3] + totalVals[i + 1][3]
            if (np.abs(sum - 3150) < 100):
                index_pos.append(i)
                index_sec_pos.append(i + 1)
                newValsX_pos.append(totalVals[i][3])
                newValsY_pos.append(totalVals[i + 1][3])


        id_neg = np.where(diffs == -1)[0]
        id_neg_sec = id_neg + 1
        newValsX_neg = []
        newValsY_neg = []
        index_neg = []
        index_sec_neg = []

        for i in id_neg:
            sum = totalVals[i][3] + totalVals[i + 1][3]
            if (np.abs(sum - 3150) < 55):
                index_neg.append(i)
                index_sec_neg.append(i + 1)
                newValsX_neg.append(totalVals[i][3])
                newValsY_neg.append(totalVals[i + 1][3])


        # selection of double events complete


        rotAngleM = (-29.72883115886924 / 180.) * np.pi
        rotMatrixM = np.array([
            [np.cos(rotAngleM), -1 * np.sin(rotAngleM)],
            [np.sin(rotAngleM), np.cos(rotAngleM)]
        ])
        rotAngleS = (-5.19673564264405 / 180.) * np.pi
        rotMatrixS = np.array([
            [np.cos(rotAngleS), -1 * np.sin(rotAngleS)],
            [np.sin(rotAngleS), np.cos(rotAngleS)]
        ])
        # translate
        translateX = -69
        translateY = -50
        # now process neg an pos vals

        # create 2 times 72 histos to bin each slice
        histos = np.zeros((2, self.__spatialBins, self.__spatialBins, 4096), dtype=np.int64)
        """
        #do a jitted version here
        @jit(nopython=True)
        def inlineCorrelator(indeces, totalVals, rotMatrixM, rotMatrixS, translateX, translateY, spatialCorr):
            sameSlice = 0

            # create 2 times 72 histos to bin each slice
            histos = np.zeros((2, 72, 72, 4096), dtype=np.int64)

            for element in indeces:
                if (element is None):
                    print("found malign element, will jump it")
                    continue
                element = int(element)
                if (element == 0):
                    print("found malign element, will jump it")
                    continue
                firstElm = totalVals[element]
                secondElm = totalVals[element + 1]
                # create rotation and translation
                if firstElm[4] == 0:
                    # Master
                    coordVec = np.array([firstElm[0] - 2048, firstElm[1] - 2048])
                    rotatedMasterCoords = rotMatrixM.dot(coordVec).T + 2048
                    firstElm[0] = rotatedMasterCoords[0]
                    firstElm[1] = rotatedMasterCoords[1]
                else:
                    # slave
                    coordVec = np.array([firstElm[0] - 2048, firstElm[1] - 2048])
                    rotatedMasterCoords = rotMatrixS.dot(coordVec).T + 2048
                    firstElm[0] = rotatedMasterCoords[0] + translateX
                    firstElm[1] = rotatedMasterCoords[1] + translateY

                if secondElm[4] == 0:
                    # Master
                    # Master
                    coordVec = np.array([secondElm[0] - 2048, secondElm[1] - 2048])
                    rotatedMasterCoords = rotMatrixM.dot(coordVec).T + 2048
                    secondElm[0] = rotatedMasterCoords[0]
                    secondElm[1] = rotatedMasterCoords[1]
                else:
                    # Slave
                    coordVec = np.array([secondElm[0] - 2048, secondElm[1] - 2048])
                    rotatedMasterCoords = rotMatrixS.dot(coordVec).T + 2048
                    secondElm[0] = rotatedMasterCoords[0] + translateX
                    secondElm[1] = rotatedMasterCoords[1] + translateY

                firstSlice = getSliceIndex(firstElm[0])
                secondSlice = getSliceIndex(secondElm[0])
                if (firstSlice != secondSlice and spatialCorr == False):
                    # we are now only interested in same slices for speeds sake
                    continue
                # check for same slice
                sameSlice += 1
                # append aat on histo depending on master or slave
                histos[int(firstElm[4])][int(firstSlice)][int(secondSlice)][int(firstElm[3])] += 1
                histos[int(secondElm[4])][int(secondSlice)][int(secondSlice)][int(secondElm[3])] += 1

                return histos
        """
        for element in np.concatenate((index_pos, index_neg)):
            if (element is None):
                print("found malign element, will jump it")
                continue
            element = int(element)
            if (element == 0):
                print("found malign element, will jump it")
                continue
            firstElm = totalVals[element]
            secondElm = totalVals[element + 1]
            # create rotation and translation
            if firstElm[4] == 0:
                # Master
                coordVec = np.array([firstElm[0] - 2048, firstElm[1] - 2048])
                rotatedMasterCoords = rotMatrixM.dot(coordVec).T + 2048
                firstElm[0] = rotatedMasterCoords[0]
                firstElm[1] = rotatedMasterCoords[1]
            else:
                # slave
                coordVec = np.array([firstElm[0] - 2048, firstElm[1] - 2048])
                rotatedMasterCoords = rotMatrixS.dot(coordVec).T + 2048
                firstElm[0] = rotatedMasterCoords[0] + translateX
                firstElm[1] = rotatedMasterCoords[1] + translateY

            if secondElm[4] == 0:
                # Master
                # Master
                coordVec = np.array([secondElm[0] - 2048, secondElm[1] - 2048])
                rotatedMasterCoords = rotMatrixM.dot(coordVec).T + 2048
                secondElm[0] = rotatedMasterCoords[0]
                secondElm[1] = rotatedMasterCoords[1]
            else:
                # Slave
                coordVec = np.array([secondElm[0] - 2048, secondElm[1] - 2048])
                rotatedMasterCoords = rotMatrixS.dot(coordVec).T + 2048
                secondElm[0] = rotatedMasterCoords[0] + translateX
                secondElm[1] = rotatedMasterCoords[1] + translateY

            firstSlice = getSliceIndex(firstElm[0], self.__spatialBins)
            secondSlice = getSliceIndex(secondElm[0], self.__spatialBins)
            if (firstSlice != secondSlice and self.__spatialCorr == False):
                #we are now only interested in same slices for speeds sake
                continue
            # append aat on histo depending on master or slave
            #something is really wrong here, check!
            histos[int(firstElm[4])][int(firstSlice)][int(secondSlice)][int(firstElm[3])] += 1
            histos[int(secondElm[4])][int(firstSlice)][int(secondSlice)][int(secondElm[3])] += 1


        #histos = inlineCorrelator(np.concatenate((index_pos, index_neg)), totalVals, rotMatrixM, rotMatrixS, translateX, translateY, self.__spatialCorr)
        self.__histos = np.add(histos, self.__histos)
        self.__histoNow = histos

    def formatHistosAsImages(self):
        #load same slice histos, bin timing new and create imag ?!
        pass