import time
import datetime

import sys
from PyQt5.QtWidgets import QDialog, QApplication,  QMainWindow, QFileDialog, QMessageBox, QSizePolicy
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QTimer
from PyQt5.uic import loadUi
import qdarkstyle

import subprocess
import time
import signal
import os

import urllib.request
import urllib

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import random

import pickle
import ntpath
import os

from Classes.lincamThread import lincamThread

import numpy as np

#routine for resolving temps
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

class AppWindow(QMainWindow):
    _container = {}
    savePath = "";
    startingTime = ""
    eventTimer = ""
    mode = "Chunked"
    chunkSize = 5.
    actualFile = ""
    running = False


    def __init__(self):
        super(AppWindow, self).__init__()
        loadUi(resource_path('QT/MainWindow.ui'), self)
        self.setFixedSize(540, 430)

        #Init Container
        self._initContainer()

        #event bindings
        self.pushButton.clicked.connect(self.fileDialogue)
        self.pushButton_3.clicked.connect(self.startRecording)
        self.pushButton_2.clicked.connect(self.stopRecording)

        #add Event timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.updateView)

        #threading
        self.lincamThread = lincamThread(self)
        self.lincamThread.signal.connect(self.threadPrinter)

    def _initContainer(self):
        pass


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

    def initStartButton(self):
        if (self.savePath != ""):
            self.pushButton_3.setEnabled(True)
        else:
            self.pushButton_3.setEnabled(False)

    def startRecording(self):
        self.running = True
        self.lockControlsSaving()
        self.label_21.setText("0")
        QMessageBox.information(self, "Lincam Started Recording", "Recording of TimeTags began")
        self.pushButton_2.setEnabled(True)
        self.pushButton_3.setEnabled(False)
        self.lineEdit_2.setEnabled(False)

        self.label_6.setText("--")
        self.startingTime = datetime.datetime.now()
        self.timer.start(500)
        self.lincamThread.start()



    def updateView(self):
        runTime = (datetime.datetime.now() - self.startingTime)
        self.label_6.setText(str(runTime).split('.')[0])

    def updateChunks(self):
       self.lincamThread.start()

    def stopRecording(self):
        self.lincamThread.quit()
        self.pushButton_2.setEnabled(False)
        QMessageBox.information(self, "Recording of TimeTags stopped", "Lincam will stop recording after this chunk")


    def finallyStopped(self):
        self.running = False
        self.enableControlsSaving()
        self.pushButton_2.setEnabled(False)
        self.pushButton_3.setEnabled(True)
        self.pushButton.setEnabled(True)
        self.timer.stop()
        self.label_21.setText("--")
        self.label_6.setText("--")
        self.lineEdit_2.setEnabled(True)

    def lockControlsSaving(self):
        self.pushButton.setEnabled(False)
        self.lineEdit_2.setEnabled(False)



    def enableControlsSaving(self):
        self.pushButton.setEnabled(True)
        self.lineEdit_2.setEnabled(True)


    def fileDialogue(self):
        self.Fileo = QFileDialog(self)
        self.Fileo.setAcceptMode(QFileDialog.AcceptOpen)
        self.Fileo.setFileMode(QFileDialog.AnyFile)

        self.Fileo.show()
        if self.Fileo.exec_():
            for item in self.Fileo.selectedFiles():
                self.savePath = item
            self.initStartButton()
            self.label_12.setText(self.savePath)


    def threadPrinter(self, msg):
        print(msg)
        #telegram Stuff
        string = urllib.parse.quote_plus(r"[TapeDrive]: " + msg)
        queryString = "https://richterbot.negative-entropy.de/messageinterface.php?pass=34m8934murx4938u&message=" + string
        contents = urllib.request.urlopen(queryString).read()


app = QApplication(sys.argv)
app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
w = AppWindow()
w.show()
sys.exit(app.exec_())