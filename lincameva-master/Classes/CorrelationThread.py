import sys
import tempfile
import subprocess
from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer
from PyQt5.QtCore import QThread, pyqtSignal
import time


class CorrelationThread(QThread):
    signal = pyqtSignal('PyQt_PyObject')
    timer = ""

    def __init__(self, parent):
        QThread.__init__(self)
        self.parent = parent
        self.runbool = True

    def run(self):
        self.signal.emit("start async corr")
        while(self.runbool):
            #print("check for uncorrelated files")
            for item in self.parent._container["fileList"]:
                if (item["status"] == 0):
                    self.signal.emit("found uncorrelated chunk: " + item["path"])
                    item["status"] = 1
                    #self.parent.updateFileList()
                    self.parent.determineCounters()
                    #begin correlation
                    starttime = time.time()
                    #check spatial bool
                    self.parent.correlator.setSpatialCorr(self.parent.checkBox_2.isChecked())
                    self.parent.correlator.loadPath(self.parent._container["inputDir"] + "/" + item["path"])
                    #self.parent.correlator.read_chunk()
                    #self.parent.correlator.Correlate()
                    self.parent.correlator.correlate()
                    self.signal.emit("Correlation completed, took " + str(time.time() - starttime) + " seconds")
                    item["status"] = 2
                    #self.parent.updateFileList()
                    self.parent.determineCounters()
                    #update histogram view
                    self.parent.printHistogram(item["path"])
                    break
            #two seconds grace after every corr, maybe not needed
            time.sleep(1)

    def start(self):
        self.runbool = True
        super().start()

    def quit(self):
        super().quit()
        self.runbool = False
        self.signal.emit("finished async correlation")