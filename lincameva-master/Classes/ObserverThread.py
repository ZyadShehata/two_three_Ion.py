import sys
import tempfile
import subprocess
from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer
from PyQt5.QtCore import QThread, pyqtSignal
import time


class ObserverThread(QThread):
    signal = pyqtSignal('PyQt_PyObject')
    timer = ""

    def __init__(self, parent):
        QThread.__init__(self)
        self.parent = parent
        self.runbool = True

    def run(self):
        print("start async watcher")
        while(self.runbool):
            #print("check for new stuff")
            self.evaluateStuff()
            time.sleep(10)

    def start(self):
        self.runbool = True
        super().start()

    def quit(self):
        super().quit()
        self.runbool = False
        print("finished async listener")

    def evaluateStuff(self):
        self.parent.evalDir()