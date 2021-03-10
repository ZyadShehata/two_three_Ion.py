import numpy as np
from photonscore.python.vault import Vault
from .DoublePhotonEvent import DoublePhotonEvent
from .EinsteinCorrection import EinsteinCorrection


class EventPreProcessor:
    # members and attributes
    __pathOne = ""  # path to file one
    __pathTwo = ""  # path to file two
    __vaultOne = None  # load vault from file one
    __vaultTwo = None  # load vault from file two

    __aatOffset = 0  # aat offset of stack two

    __generalincrement = 0  # save last position on stack one
    __generalincrementsize = 100  # process n entries from stack one at a time
    __generalcoincidencetime = 20  # n * 10ns (aat clock)

    __confidenceSlope = -1.  # slope for the confidence checker for double event tracking
    __confidenceOffsetOne = 4095. + 5.  # upper offset for confidence interval
    __confidenceOffsetTwo = 4095. - 5.  # lower offset for confidence interval

    __TacConversionOne = 1.  # Conversion from TAC integer to real time
    __TacConversionTwo = 1.  # Conversion from TAC integer to real time

    __TacRebinningFactor = 1.  # For einstein correction, if detection window is changed

    __indexVaultOne = 0  # last processed index on vault one
    __indexVaultTwo = 0  # last processed index on vault two

    __vaultOneEnd = False  # Flag for reaching end of vault one
    __vaultTwoEnd = False  # Flag for reaching end of vault two

    __doubleEvents = []  # completely processed double events are stored here

    __detectorMasterNetwork = ""  # weights for network for einstein correction
    __detectorSlaveNetwork = ""  # weights for network for einstein correction

    # init stuff
    def __init__(self):
        self.__detectorMasterNetwork = np.fromfile("./clbr/PX29110.double", dtype=np.float64)
        self.__detectorSlaveNetwork = np.fromfile("./clbr/PX29111.double", dtype=np.float64)

    def __is_init(self):

        if self.__pathOne == "":
            return False
        if self.__pathTwo == "":
            return False
        if self.__vaultOne is None:
            return False
        if self.__vaultTwo is None:
            return False

        return True

    # setter and initialization
    def set_path_one(self, newpath):
        self.__pathOne = newpath

    def set_path_two(self, newpath):
        self.__pathTwo = newpath

    def set_aat_offset(self, newoffset):
        self.__aatOffset = newoffset

    def set_increment_size(self, newincrement):
        self.__generalincrementsize = newincrement

    def set_coincidence_window_multiplicator(self, newcoincidence):
        self.__generalcoincidencetime = newcoincidence

    def set_confidence_slope(self, slope):
        self.__confidenceSlope = slope

    def set_confidence_offset_one(self, offset):
        self.__confidenceOffsetOne = offset

    def set_confidence_offset_two(self, offset):
        self.__confidenceOffsetTwo = offset

    def load_vaults(self):
        self.__vaultOne = Vault(self.__pathOne)
        self.__vaultTwo = Vault(self.__pathTwo)
        self.__vaultOneEnd = False
        self.__vaultTwoEnd = False
        self.__generalincrement = 0

    def reset(self):
        self.__pathOne = ""  # path to file one
        self.__pathTwo = ""  # path to file two
        self.__vaultOne = None  # load vault from file one
        self.__vaultTwo = None  # load vault from file two

        self.__aatOffset = 0  # aat offset of stack two

        self.__generalincrement = 0  # save last position on stack one
        self.__generalincrementsize = 10000  # process n entries from stack one at a time
        self.__generalcoincidencetime = 20  # n * 10ns (aat clock)

        self.__confidenceSlope = -0.3  # slope for the confidence checker for double event tracking
        self.__confidenceOffsetOne = 2.  # upper offset for confidence interval
        self.__confidenceOffsetTwo = 1.  # lower offset for confidence interval

        self.__TacConversionOne = 1.  # Conversion from TAC integer to real time
        self.__TacConversionTwo = 1.  # Conversion from TAC interger to real time

        self.__TacRebinningFactor = 1.  # For einstein correction, if detection window is changed

        self.__indexVaultOne = 0  # last processed index on vault one
        self.__indexVaultTwo = 0  # last processed index on vault two

        self.__vaultOneEnd = False  # Flag for reaching end of vault one
        self.__vaultTwoEnd = False  # Flag for reaching end of vault two

        self.__doubleEvents = []  # completely processed double events are stored here

    # main functions
    def process_increment(self):
        # check for all parameters
        if not self.__is_init():
            return False
        # chec if not finished
        if self.is_finished():
            return False

        startIncrement = self.__generalincrement * self.__generalincrementsize
        stopIncrement = ((self.__generalincrement + 1) * self.__generalincrementsize) - 1
        # check for end of vaults
        stopIncrement = min(stopIncrement, self.__vaultOne.data.photons.x.shape[0], self.__vaultTwo.data.photons.x.shape[0])

        # set end of vault flags
        if stopIncrement == self.__vaultOne.data.photons.x.shape[0]:
            self.__vaultOneEnd = True
        if stopIncrement == self.__vaultTwo.data.photons.x.shape[0]:
            self.__vaultTwoEnd = True

        # get all events from stack one lying in increment
        dataOne = {
            "x": self.__vaultOne.data.photons.x[startIncrement:stopIncrement],
            "y": self.__vaultOne.data.photons.y[startIncrement:stopIncrement],
            "dt": self.__vaultOne.data.photons.dt[startIncrement:stopIncrement],
            "aat": self.__vaultOne.data.photons.aat[startIncrement:stopIncrement],
            "a": self.__vaultOne.data.photons.a[startIncrement:stopIncrement],
        }
        dataTwo = {
            "x": self.__vaultTwo.data.photons.x[startIncrement:stopIncrement],
            "y": self.__vaultTwo.data.photons.y[startIncrement:stopIncrement],
            "dt": self.__vaultTwo.data.photons.dt[startIncrement:stopIncrement],
            "aat": self.__vaultTwo.data.photons.aat[startIncrement:stopIncrement],
            "a": self.__vaultTwo.data.photons.a[startIncrement:stopIncrement]
        }
        # iterate over all events
        for i in range(len(dataOne["x"])):
            # update stack position
            self.__indexVaultOne = i
            # construct index span for doubled coincindence time in stack two
            temporalAat = dataOne["aat"][i]
            aatLow = int((temporalAat - self.__generalcoincidencetime) + self.__aatOffset)
            aatHigh = int((temporalAat + self.__generalcoincidencetime) + self.__aatOffset)
            # search needed events in stack two
            possibleDoubleEvents = []
            j = self.__indexVaultTwo
            while True:
                #check for ned of list
                if j >= dataTwo["aat"].shape[0]:
                    break
                # check aat
                self.__indexVaultTwo = j
                stackTwoAat = dataTwo["aat"][j]
                if stackTwoAat > aatHigh:
                    break
                if aatLow <= stackTwoAat <= aatHigh:
                    possibleDoubleEvents.append(j)
                j += 1

            # all possible double events found, try to process them
            if len(possibleDoubleEvents) == 0:
                continue

            # iterate over stack two elements and check for double event by using slope
            for k in range(len(possibleDoubleEvents)):
                if self.__check_for_double_timing(dataOne["dt"][i], dataTwo["dt"][k]):
                    # create einstein correction for both values
                    EinsteinBase = EinsteinCorrection.correct_single(dataOne["x"][i], dataOne["y"][i], dataOne["a"][i],
                                                                     self.__detectorMasterNetwork) * 4095. * self.__TacRebinningFactor
                    EinsteinSecond = EinsteinCorrection.correct_single(dataTwo["x"][k], dataTwo["y"][k],
                                                                       dataTwo["a"][k],
                                                                       self.__detectorSlaveNetwork) * 4095. * self.__TacRebinningFactor
                    EinsteinDifference = EinsteinBase - EinsteinSecond
                    # decide case (which event was first)
                    # do case sensitive correction
                    if dataOne["aat"][i] - dataTwo["aat"][k] < 0:
                        correctTimingOne = (dataOne["dt"][i] - EinsteinDifference) * self.__TacConversionOne
                        correctTimingTwo = (dataTwo["dt"][k] + EinsteinDifference) * self.__TacConversionTwo
                    else:
                        correctTimingOne = (dataOne["dt"][i] + EinsteinDifference) * self.__TacConversionOne
                        correctTimingTwo = (dataTwo["dt"][k] - EinsteinDifference) * self.__TacConversionTwo

                    # create double event and push to storage
                    temporalDoubleEvent = DoublePhotonEvent(dataOne["x"][i], dataOne["y"][i], dataTwo["x"][k],
                                                            dataTwo["y"][k], correctTimingOne[0], correctTimingTwo[0])
                    self.__doubleEvents.append(temporalDoubleEvent)

                    # exit, we only need one second event
                    break

        # higher incremental position
        self.__generalincrement += 1

    # check for doule event validity by confidence interval of delta t
    def __check_for_double_timing(self, deltaOne, deltaTwo):
        #return True
        return self.__confidenceSlope * deltaOne + self.__confidenceOffsetOne > deltaTwo > self.__confidenceSlope * deltaOne + self.__confidenceOffsetTwo

    def process_all_increments(self):
        while not self.is_finished():
           self.process_increment()

    def get_double_events(self, resetbuffer=False):
        doubleList = self.__doubleEvents.copy()
        if resetbuffer:
            self.__doubleEvents = []
        return doubleList

    def is_finished(self):
        return self.__vaultOneEnd or self.__vaultTwoEnd
