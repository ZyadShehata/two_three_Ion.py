import numpy as np
from numba import jit, cuda, njit

class Correlator:

    __mode = "Numpy" #Numpy, Loop or CUDA
    __jit = False
    __filepath = "" #path to rawdata binary
    __chunksize = -1 # size of chunks, -1 for whole file, other currently not supported
    __offset = 0 # start at beginning

    __length = 10000
    __steps = 10

    __channels = []
    __times = []

    __histogram = []
    __histogramAxes = []

    def __init__(self):
        pass

    def setMode(self, newMode):
        if newMode not in ["Numpy", "Loop", "CUDA"]:
            raise ValueError("Unsupported Mode. Only Numpy, Loop and CUDA allowed!")
        self.__mode = newMode
    def setJit(self, newJitBool):
        self.__jit = newJitBool

    def resetHistograms(self):
        self.__histogram = []
        self.__histogramAxes = []

    def getHistograms(self):
        return self.__histogram, self.__histogramAxes

    def setFilePath(self, newPath):
        self.__filepath = newPath

    def setLengthsAndSteps(self, Length, Steps):
        self.__length = Length
        self.__steps = Steps

    def loadTestData(self):
        self.__times = np.array([2, 4, 6, 8, 12, 13, 16, 19, 26, 28, 30, 35, 40, 40, 43, 44, 47, 49, 50], dtype=np.int64)
        self.__channels = np.array([1, 1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2], dtype=np.int16)

    def __build_boundaries(self, zeroBin = False):
        if (zeroBin):
            return np.linspace(-0.5 * self.__steps, self.__length + 0.5 * self.__steps, num=(int(self.__length / self.__steps) + 2))
        return np.linspace(0.5 * self.__steps, self.__length + 0.5 * self.__steps, num=(int(self.__length / self.__steps) + 1))

    def Correlate(self):
        if (self.__jit == False):
            if (self.__mode == "Numpy"):
                totalHist = self.__correlateNumpy()
            if (self.__mode == "Loop"):
                totalHist = self.__correlateLoop()
        else:
            if (self.__mode == "Numpy"):
                totalHist = self.__correlateNumpyJit(np.array(self.__channels), np.array(self.__times), np.concatenate((np.flipud(-1 * self.__build_boundaries()), self.__build_boundaries(True)[1:])))
            if (self.__mode == "Loop"):
                totalHist = self.__correlateLoopJit(np.array(self.__channels), np.array(self.__times), np.concatenate((np.flipud(-1 * self.__build_boundaries()), self.__build_boundaries(True)[1:])))
        #Cuda is jitted everytime!
        if (self.__mode == "CUDA"):
            totalHist = self.__correlateCuda()

        #totaloValues = np.concatenate([histTwo[0][1:], histOne[0][:-1]])
        #totaloAxis = np.concatenate([histTwo[1][1:-1], histOne[1][1:-1]])

        totaloValues = totalHist[0][1:-1]
        totaloAxis = totalHist[1][1:-1]

        if (len(self.__histogram) == 0):
            self.__histogram = totaloValues
            self.__histogramAxes = totaloAxis
        else:
            self.__histogram = np.add(self.__histogram, totaloValues)






    def read_chunk(self):  # chunksize
        # consider 40 byte header: first pointer has to be chosen accordingly
        neededOffset = self.__offset
        if (neededOffset == 0):
            neededOffset = 40
        with open(self.__filepath, 'rb') as f:
            f.seek(neededOffset, 0)
            dt = np.dtype([('timestamp', np.int64), ('channel', np.int16)])
            read_file = np.fromfile(f, dtype=dt, count=self.__chunksize)
            channels = read_file['channel']
            times = read_file['timestamp']
            if (self.__chunksize != -1):
                self.__offset += self.__chunksize
        f.close()
        self.__channels = channels
        self.__times = times

    def __correlateNumpy(self):
        channeldiffs = np.diff(self.__channels)
        idx_pos = np.where(channeldiffs == 1)[0]
        idx_pos_two = idx_pos + 1
        idx_neg = np.where(channeldiffs == -1)[0]
        idx_neg_two = idx_neg + 1
        val_pos = np.subtract(self.__times[idx_pos_two], self.__times[idx_pos])
        val_neg = np.subtract(self.__times[idx_neg_two], self.__times[idx_neg])
        # create bins
        bins = self.__build_boundaries()
        binsWithZero = self.__build_boundaries(True)
        totalBins = np.concatenate((np.flipud(-1 * bins), binsWithZero[1:]))
        totalVals = np.concatenate((-1 * val_neg, val_pos))
        totalHist = np.histogram(totalVals, totalBins)
        return totalHist

    def __correlateLoop(self):
        val_pos = []
        val_neg = []
        for i in range(len(self.__times)):
            if (i == 0):
                continue
            if (self.__channels[i] - self.__channels[i-1] == 1):
                val_pos.append(self.__times[i] - self.__times[i-1])
            if (self.__channels[i] - self.__channels[i-1] == -1):
                val_neg.append(self.__times[i] - self.__times[i - 1])
        bins = self.__build_boundaries()
        binsWithZero = self.__build_boundaries(True)
        totalBins = np.concatenate((np.flipud(-1 * bins), binsWithZero[1:]))
        totalVals = np.concatenate((-1 * np.array(val_neg), val_pos))
        totalHist = np.histogram(totalVals, totalBins)
        #histPos = np.histogram(val_pos, self.__build_boundaries(True))
        #histNeg = np.histogram(-1 * np.array(val_neg), np.flipud(-1 * bins))
        return totalHist


    def __correlateNumpyParallel(self):
        raise ValueError("Feature temporarily discarded")
        pass

    #calls Cuda Kernel
    def __correlateCuda(self):
        #memory stuff
        cudaChannels = cuda.to_device(self.__channels)
        cudaTimes = cuda.to_device(self.__times)

        baseHisto = np.zeros((2, self.__length), dtype=np.int32)
        cudaBaseHisto = cuda.to_device(baseHisto)

        #threadgrid stuff
        threadsperblock = 256
        blockspergrid = self.__channels.shape[0] // 256 + 1
        self.__cudaCorrelationKernel[blockspergrid, threadsperblock](cudaChannels, cudaTimes, cudaBaseHisto)

        #fetch data from mem
        processedHisto = cudaBaseHisto.copy_to_host()
        #postprocessing


        return processedHisto


    #JIT SECTION
    @staticmethod
    @jit(nopython=True)
    def __correlateNumpyJit(channels, times, totalboundaries):
        channeldiffs = np.diff(channels)
        idx_pos = np.where(channeldiffs == 1)[0]
        idx_pos_two = idx_pos + 1
        idx_neg = np.where(channeldiffs == -1)[0]
        idx_neg_two = idx_neg + 1
        val_pos = np.subtract(times[idx_pos_two], times[idx_pos])
        val_neg = np.subtract(times[idx_neg_two], times[idx_neg])
        # create bins
        totalVals = np.concatenate((-1 * val_neg, val_pos))
        totalHist = np.histogram(totalVals, totalboundaries)
        return totalHist

    @staticmethod
    @jit(nopython=True)
    def __correlateLoopJit(channels, times, totalboundaries):
        val_pos = []
        val_neg = []
        for i in range(len(times)):
            if (i == 0):
                continue
            if (channels[i] - channels[i - 1] == 1):
                val_pos.append(times[i] - times[i - 1])
            if (channels[i] - channels[i - 1] == -1):
                val_neg.append(times[i] - times[i - 1])
        totalVals = np.concatenate((-1 * np.array(val_neg), np.array(val_pos)))
        totalHist = np.histogram(totalVals, totalboundaries)
        return totalHist

    # CUDA Section
    @staticmethod
    @cuda.jit
    def __cudaCorrelationKernel(channels, times, basehisto):
        #check position in thread grid
        i = cuda.grid(1)
        #determine max pos
        maxDim = channels.shape[0] - 1
        #we do not process zero timetag
        if (i == 0):
            return
        #check if outside of data
        if (i > maxDim):
            return
        #we are only interesetd in cross correlation
        if (channels[i] - channels[i - 1] == 0):
            return
        #if we reached here, we can determine timediff in basic TDC bins
        timediff = times[i] - times[i - 1]
        #check if timing diff lies inside histo
        if (timediff > basehisto.shape[1] - 1 or timediff < 0):
            return
        #ok, now fill pos and neg histo, only call atomics if really needed
        if (channels[i] - channels[i - 1] == 1):
            cuda.atomic.add(basehisto, [0][timediff], 1)
        if (channels[i] - channels[i - 1] == -1):
            cuda.atomic.add(basehisto, [1][timediff], 1)
