import numpy as np

#Mainly three modes of operation
# 1. Two detectors at the same place (redo APD experiment)
# 2. Two detectors at different positions, take short coincidence window
# 3. Tempral correlation

class Correlator:

    eventStack = []
    lastProcessedEvent = 0

    mode = "samePosition"

    histogram = []


    def __init__(self):
        pass

    def loadnewevents(self, newEvents):
        self.eventStack = newEvents
        self.lastProcessedEvent = 0