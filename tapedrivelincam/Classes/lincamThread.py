import sys
import tempfile
import subprocess
from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer
from PyQt5.QtCore import QThread, pyqtSignal
import time

import os


class lincamThread(QThread):
    signal = pyqtSignal('PyQt_PyObject')
    timer = ""

    def __init__(self, parent):
        QThread.__init__(self)
        self.parent = parent
        self.runbool = True

    def runLinCam(self, seconds, outputFile):
        proc = subprocess.Popen([r"C:\Users\Photonscore\Desktop\erlangen-readout\erlangen_readout.exe", r"--log=stdout",
                                 r"--measure-for=" + str(seconds), r"--master-tac-window=4", r"--slave-tac-window=4",
                                 r"--master-tac-bias=2000", r"--slave-tac-bias=2000",
                                 r"c:\Users\Photonscore\Desktop\erlangen-readout\PX29110_master.profile",
                                 r"c:\Users\Photonscore\Desktop\erlangen-readout\PX29109_slave.profile",
                                 str(outputFile)], stdin=subprocess.PIPE,
                                stdout=sys.stdout.fileno(), stderr=sys.stdout.fileno(),
                                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP, shell=True)
        return proc.wait()

    def run(self):
        while(self.runbool):
            self.signal.emit("start async lincam recording")
            if(self.runbool):
                #naming stuff
                filename = self.parent.savePath
                if (int(self.parent.label_21.text()) != 0):
                    os.rename(filename.replace(".photons", "") + self.parent.label_21.text() + ".photons.temp",
                             filename.replace(".photons", "") + self.parent.label_21.text() + ".photons")

                self.parent.label_21.setText(str(int(self.parent.label_21.text()) + 1))
                filename = filename.replace(".photons", "") + self.parent.label_21.text() + ".photons.temp"
                self.parent.actualFile = filename

                if(self.runLinCam(float(self.parent.lineEdit_2.text())*60, filename) == 0):
                    self.signal.emit("Chunk was recorded correctly")
                else:
                    self.signal.emit("Error recording Chunk")

                time.sleep(1)
        self.signal.emit("recording was disabled successfully")
        if (int(self.parent.label_21.text()) != 0):
            filename = self.parent.savePath
            os.rename(filename.replace(".photons", "") + self.parent.label_21.text() + ".photons.temp",
                      filename.replace(".photons", "") + self.parent.label_21.text() + ".photons")
        self.parent.finallyStopped()

    def start(self):
        self.runbool = True
        super().start()

    def quit(self):
        super().quit()
        self.runbool = False
