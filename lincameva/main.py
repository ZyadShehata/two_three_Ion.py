import sys
from PyQt5.QtWidgets import QDialog, QApplication,  QMainWindow, QFileDialog, QMessageBox, QSizePolicy
from PyQt5.QtGui import QIcon
from PyQt5.uic import loadUi
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QThread, pyqtSignal
import qdarkstyle

import urllib.request
import urllib


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from Classes.PlotCanvas import PlotCanvas
from Classes.ObserverThread import ObserverThread
from Classes.CorrelationThread import CorrelationThread
from Classes.DoubleImageCanvas import DoubleImageCanvas
from Classes.HybridCanvas import HybridCanvas
from Classes.DoublePlotCanvas import DoublePlotCanvas

from include.Correlations import Correlator
from include.CorrelationsLin import CorrelatorLin

import matplotlib.pyplot as plt

import random

import pickle
import ntpath
import os
import glob
import time

import numpy as np

plt.style.use('dark_background')

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



    def __init__(self):
        super(AppWindow, self).__init__()
        loadUi(resource_path('QT/MainWindow.ui'), self)

        #Init Container
        self._initContainer()

        #Init Canvas g1
        self.PlotCanvasNormed = DoubleImageCanvas(self.groupBox_2, width=5, height=4)
        self.PlotCanvasNormed.move(300,300)
        self.NavBar = NavigationToolbar(self.PlotCanvasNormed, self.groupBox_2)
        self.verticalLayout_7.addWidget(self.PlotCanvasNormed)
        self.verticalLayout_7.addWidget(self.NavBar)

        # Init Canvas G2
        self.PlotCanvasG2 = DoubleImageCanvas(self.groupBox_2, width=5, height=4)
        self.PlotCanvasG2.move(300, 300)
        self.NavBar = NavigationToolbar(self.PlotCanvasG2, self.groupBox_2)
        self.verticalLayout_9.addWidget(self.PlotCanvasG2)
        self.verticalLayout_9.addWidget(self.NavBar)

        #init Canvas g2
        self.PlotCanvasg2 = HybridCanvas(self.groupBox_2, width=5, height=4)
        self.PlotCanvasg2.move(300, 300)
        self.NavBar = NavigationToolbar(self.PlotCanvasg2, self.groupBox_2)
        self.verticalLayout_11.addWidget(self.PlotCanvasg2)
        self.verticalLayout_11.addWidget(self.NavBar)

        #init Canvas t=0
        self.PlotCanvasT0 = DoublePlotCanvas(self.groupBox_2, width=5, height=4)
        self.PlotCanvasT0.move(300, 300)
        self.NavBar = NavigationToolbar(self.PlotCanvasT0, self.groupBox_2)
        self.verticalLayout_24.addWidget(self.PlotCanvasT0)
        self.verticalLayout_24.addWidget(self.NavBar)

        # init Canvas t=0
        self.PlotCanvasdx = DoublePlotCanvas(self.groupBox_2, width=5, height=4)
        self.PlotCanvasdx.move(300, 300)
        self.NavBar = NavigationToolbar(self.PlotCanvasdx, self.groupBox_2)
        self.verticalLayout_26.addWidget(self.PlotCanvasdx)
        self.verticalLayout_26.addWidget(self.NavBar)

        #init Canvas Spatial
        self.PlotCanvasSpatial = DoubleImageCanvas(self.groupBox_2, width=5, height=4)
        self.PlotCanvasSpatial.move(300, 300)
        self.NavBar = NavigationToolbar(self.PlotCanvasSpatial, self.groupBox_2)
        self.verticalLayout_13.addWidget(self.PlotCanvasSpatial)
        self.verticalLayout_13.addWidget(self.NavBar)

        #init Canvas merged
        self.PlotCanvasMerged = DoubleImageCanvas(self.groupBox_2, width=5, height=4)
        self.PlotCanvasMerged.move(300, 300)
        self.NavBar = NavigationToolbar(self.PlotCanvasMerged, self.groupBox_2)
        self.verticalLayout_29.addWidget(self.PlotCanvasMerged)
        self.verticalLayout_29.addWidget(self.NavBar)

        #add Event Listeners
        self.browseInput.clicked.connect(self.fileDialogInput)
        self.pushButton_3.clicked.connect(self.evalDirFirst)
        self.buttonProcess.clicked.connect(self.startWorking)
        self.pushButton.clicked.connect(self.stopWorking)
        self.browseInput_2.clicked.connect(self.saveDialogInput)
        self.pushButton_2.clicked.connect(self.saveHistogram)
        self.pushButton_5.clicked.connect(self.saveChunksInput)
        self.pushButton_7.clicked.connect(self.loadHisto)
        self.pushButton_8.clicked.connect(self.saveG1TotInputFileDialogue)
        self.pushButton_9.clicked.connect(self.saveTotalG1Action)
        self.pushButton_6.clicked.connect(self.doSpatialCorr)
        self.pushButton_10.clicked.connect(self.updateFileList)
        self.pushButton_11.clicked.connect(self.refreshg2Histo)
        self.pushButton_12.clicked.connect(self.refreshg2Histo)
        self.pushButton_13.clicked.connect(self.saveChunkedG1)
        self.pushButton_14.clicked.connect(self.doMerging)

        #thread classes
        self.observerThread = ObserverThread(self)
        self.observerThread.signal.connect(self.threadPrinter)

        self.correlationThread = CorrelationThread(self)
        self.correlationThread.signal.connect(self.threadPrinter)

        #Load Correlator
        self.correlator = CorrelatorLin()

        #set Image Canvas
        self.PlotCanvasNormed.plot(np.zeros((100,100)), np.zeros((100,100)), np.zeros((100,100)), np.zeros((100,100)))
        self.PlotCanvasG2.plot(np.zeros((100,100)), np.zeros((100,100)), np.zeros((100,100)), np.zeros((100,100)))
        self.PlotCanvasg2.plot(np.zeros((64,144)), np.zeros((64,144)), np.array([range(int(self.lineEdit_3.text())), np.zeros(int(self.lineEdit_3.text()))]), np.array([range(int(self.lineEdit_3.text())), np.zeros(int(self.lineEdit_3.text()))]), "g2 Master", "g2 Slave", "G1 Master", "G1 Slave")
        self.PlotCanvasSpatial.plot(np.zeros((100,100)), np.zeros((100,100)), np.zeros((100,100)), np.zeros((100,100)), "Spatial Master Raw", "Spatial Slave Raw", "Spatial Master normed", "Spatial Slave normed")
        self.PlotCanvasT0.plot(np.array([range(int(self.lineEdit_3.text())), np.zeros(int(self.lineEdit_3.text()))]), np.array([range(int(self.lineEdit_3.text())), np.zeros(int(self.lineEdit_3.text()))]), "Master Tau 0", "Slave Tau 0")
        self.PlotCanvasdx.plot(np.array([range(int(self.lineEdit_3.text())), np.zeros(int(self.lineEdit_3.text()))]), np.array([range(int(self.lineEdit_3.text())), np.zeros(int(self.lineEdit_3.text()))]), "Position One", "Position Two")
        self.PlotCanvasMerged.plot(np.zeros((100,100)), np.zeros((100,100)), None, None, "G2", "g2")


        #init totalG1Arr
        self._totalG1Array = np.zeros((2,512,512), dtype=np.int32)

        #advanced UI Stuff
        for i in range(1,int(self.lineEdit_3.text())):
            self.comboBox.addItem(str(i))




    #real Aux Stuff
    def path_leaf(self, path):
        head, tail = ntpath.split(path)
        return tail or ntpath.basename(head)

    #Auxiliary and Application functions
    def disableUi(self):
        self.browseInput.setEnabled(False)
        self.pushButton_3.setEnabled(False)
        self.buttonProcess.setEnabled(False)
        self.pushButton.setEnabled(True)
        self.pushButton_4.setEnabled(False)


    def enableUi(self):
        self.browseInput.setEnabled(True)
        self.pushButton_3.setEnabled(True)
        self.buttonProcess.setEnabled(True)
        self.pushButton.setEnabled(False)
        self.pushButton_4.setEnabled(True)


    #ContainerStuff
    def _initContainer(self):
        self._container["rawData"] = np.zeros((2, 20), dtype=np.float64)
        self._container["rawData"][0] = np.array(range(20))
        self._container["inputDir"] = ""
        self._container["beforeFiles"] = {}
        self._container["fileList"] = []


    #Define Form Logic
    def openInfo(self):
        QMessageBox.about(self, 'Info', "A small program that listens to a folder for files created by TapeDrive to correlate them numerically in an efficient way \n powered by Stefano")

    def fileDialogInput(self):
        self.Fileo = QFileDialog(self)
        self.Fileo.setAcceptMode(QFileDialog.AcceptOpen)
        self.Fileo.setFileMode(QFileDialog.Directory)

        self.Fileo.show()
        if self.Fileo.exec_():
            self._container["inputDir"] = self.Fileo.selectedFiles()[0]
            self.label_5.setText(self._container["inputDir"])
            self.pushButton_3.setEnabled(True)

    def saveDialogInput(self):
        self.Fileo = QFileDialog(self)
        self.Fileo.setAcceptMode(QFileDialog.AcceptOpen)
        self.Fileo.setFileMode(QFileDialog.AnyFile)

        self.Fileo.show()
        if self.Fileo.exec_():
            self.label_4.setText(self.Fileo.selectedFiles()[0])

    def saveChunksInput(self):
        self.Fileo = QFileDialog(self)
        self.Fileo.setAcceptMode(QFileDialog.AcceptOpen)
        self.Fileo.setFileMode(QFileDialog.Directory)

        self.Fileo.show()
        if self.Fileo.exec_():
            self.label_6.setText(self.Fileo.selectedFiles()[0])


    def saveG1TotInputFileDialogue(self):
        self.Fileo = QFileDialog(self)
        self.Fileo.setAcceptMode(QFileDialog.AcceptOpen)
        self.Fileo.setFileMode(QFileDialog.AnyFile)

        self.Fileo.show()
        if self.Fileo.exec_():
            self.label_21.setText(self.Fileo.selectedFiles()[0])

    def saveTotalG1Action(self):
        if(self.label_21.text() != "NotSet"):
            np.save(self.label_21.text(), self._totalG1Array)
        else:
            QMessageBox.about(self, 'No Path', "Choose Path first")

    def loadHisto(self):
        self.Fileo = QFileDialog(self)
        self.Fileo.setAcceptMode(QFileDialog.AcceptOpen)
        self.Fileo.setFileMode(QFileDialog.AnyFile)

        self.Fileo.show()
        if self.Fileo.exec_():
            temporalHist = np.load(self.Fileo.selectedFiles()[0])
            self.correlator.loadAndAppendHisto(temporalHist)
            self.printHistogram()

    def saveHistogram(self):
        if (self.label_4.text() == "NotSet"):
            QMessageBox.about(self, 'No Path', "Choose Path first")
            return 0
        np.save(self.label_4.text(), self.correlator.getHistos())
        QMessageBox.about(self, 'Success', "Saving was successful")

    def evalDir(self):
        after = dict([(f, None) for f in glob.glob(self._container["inputDir"] + "/*.photons")])
        added = [f for f in after if not f in self._container["beforeFiles"]]
        for item in added:
            self._container["fileList"].append(
                {
                    "path": self.path_leaf(item),
                    "status": 0
                }
            )
        self._container["beforeFiles"] = after
        #add added to storage with right flag
        #self.updateFileList()

    def evalDirFirst(self):
        self.evalDir()
        self.updateFileList()

    def updateFileList(self):
        self.listWidget.clear()
        counter = 0
        counterProcessed = 0
        counterWaiting = 0
        for item in self._container["fileList"]:
            counter += 1
            if (int(item["status"]) == 2):
                counterProcessed += 1
            else:
                counterWaiting += 1
            self.listWidget.addItem(item["path"])
            listItem = self.listWidget.findItems(item["path"], Qt.MatchExactly)[0]
            listItem.setCheckState(int(item["status"]))

        self.label.setText(str(counter))
        self.label_7.setText(str(counterProcessed))
        self.label_10.setText(str(counterWaiting))

    def determineCounters(self):
        counter = 0
        counterProcessed = 0
        counterWaiting = 0
        for item in self._container["fileList"]:
            counter += 1
            if (int(item["status"]) == 2):
                counterProcessed += 1
            else:
                counterWaiting += 1

        self.label.setText(str(counter))
        self.label_7.setText(str(counterProcessed))
        self.label_10.setText(str(counterWaiting))

    def listItemScope(self, item):
        buttonReply =  QMessageBox.information(self, "List Detail", "Pfad: " + item.setText(), QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Cancel)
        if buttonReply == QMessageBox.Discard:
            item.listWidget().takeItem(item.listWidget().row(item))

    def startWorking(self):
        self.disableUi()

        #update bin stuff
        self.comboBox.clear()
        for i in range(1,int(self.lineEdit_3.text())):
            self.comboBox.addItem(str(i))

        self.correlator.setSpatialBins(int(self.lineEdit_3.text()))
        #init correlator, length and steps cannot be changed without reset... TBD
        #self.correlator.setLengthsAndSteps(int(self.lineEdit.setText()), int(self.lineEdit_2.setText()))
        #start overwatcher thread
        self.observerThread.start()
        #start corr thread
        self.correlationThread.start()


    def stopWorking(self):
        self.enableUi()
        #stop both threads
        self.observerThread.quit()
        self.correlationThread.quit()


    def resetAll(self):
        self.correlator.resetHisto()
        self.printHistogram()
        #now also reset canvas
        #TBD

    def threadPrinter(self, msg):
        print(msg)
        # telegram Stuff
        string = urllib.parse.quote_plus(r"[Correlator]: " + msg)
        queryString = "https://richterbot.negative-entropy.de/messageinterface.php?pass=34m8934murx4938u&message=" + string
        if (self.checkBox.isChecked()):
            contents = urllib.request.urlopen(queryString).read()

    def doSpatialCorr(self):
        #QMessageBox.about(self, 'No Done Yet', "Stefan was to lazy")

        def getSameChannels(histo, detector=0):
            len = histo.shape[1]
            out = np.zeros((len, histo.shape[3]), dtype=np.int32)
            for i in range(len):
                out[i] = histo[detector][i][i]
            return out

        def getCrossChannels(histo, startChannel, detector=0):
            len = histo.shape[1]
            out = np.zeros((len, histo.shape[3]), dtype=np.int32)
            for i in range(len):
                out[i] = histo[detector][startChannel][i]
            return out

        histoNow, histAcc = self.correlator.getHistos()



        # test plotting
        rawVals = getCrossChannels(histAcc, int(self.comboBox.currentText()), 0)
        rawValsM = rawVals
        # rebin second axis
        # how many rebins?
        rebinner = 64  # factorial of 2 (4, 8, 16, 32, ...)!
        rebinned = np.zeros((rawVals.shape[0], int(rawVals.shape[1] / rebinner)), dtype=np.int32)
        for i in range(rawVals.shape[0]):
            rebinned[i] = np.sum(rawVals[i].reshape(-1, rebinner), axis=1)
        histMasterAcc = rebinned.T

        # test plotting
        rawVals = getCrossChannels(histAcc, int(self.comboBox.currentText()), 1)
        rawValsS = rawVals
        # rebin second axis
        # how many rebins?
        rebinner = 64  # factorial of 2 (4, 8, 16, 32, ...)!
        rebinned = np.zeros((rawVals.shape[0], int(rawVals.shape[1] / rebinner)), dtype=np.int32)
        for i in range(rawVals.shape[0]):
            rebinned[i] = np.sum(rawVals[i].reshape(-1, rebinner), axis=1)
        histSlaveAcc = rebinned.T

        #do normed stuff here

        # add plotting of binned G1 here!!
        imagBinnedMaster, imagBinnedSlave = self.correlator.getBinnedG1()

        normValMaster = np.sum(imagBinnedMaster) ** 2
        normValSlave = np.sum(imagBinnedSlave) ** 2

        normValHistM = np.sum(histMasterAcc)
        normValHistS = np.sum(histSlaveAcc)

        rebinner = 64  # factorial of 2 (4, 8, 16, 32, ...)!
        rebinned = np.zeros((rawValsM.shape[0], int(rawValsM.shape[1] / rebinner)), dtype=np.float64)
        for i in range(rawValsM.shape[0]):
            totaloMean = (imagBinnedMaster[1][int(self.comboBox.currentText())] * imagBinnedMaster[1][i]) * 0.1
            rebinned[i] = (1. / (totaloMean / normValMaster)) * (
                        np.sum(rawVals[i].reshape(-1, rebinner), axis=1) / normValHistM)
        normedMaster = rebinned.T

        rebinner = 64  # factorial of 2 (4, 8, 16, 32, ...)!
        rebinned = np.zeros((rawValsS.shape[0], int(rawValsS.shape[1] / rebinner)), dtype=np.float64)
        for i in range(rawValsS.shape[0]):
            totaloMean = (imagBinnedSlave[1][int(self.comboBox.currentText())] * imagBinnedSlave[1][i]) * 0.1
            rebinned[i] = (1. / (totaloMean / normValSlave)) * (
                    np.sum(rawVals[i].reshape(-1, rebinner), axis=1) / normValHistS)
        normedSlave = rebinned.T


        self.PlotCanvasSpatial.plot(histMasterAcc, histSlaveAcc, normedMaster, normedSlave, "Spatial Master Raw", "Spatial Slave Raw", "Spatial Master normed", "Spatial Slave normed", int(self.lineEdit_6.text()), int(self.lineEdit_7.text()))

    def doMerging(self):

        def getSameChannels(histo, detector=0):
            len = histo.shape[1]
            out = np.zeros((len, histo.shape[3]), dtype=np.int32)
            for i in range(len):
                out[i] = histo[detector][i][i]
            return out

        timeScale = float(self.lineEdit_8.text())*1e-9
        #create neededBins
        bins = np.arange(-40e-9, 50e-9, timeScale)

        histoNow, histAcc = self.correlator.getHistos()

        histoMaster = getSameChannels(histAcc, 0)
        histoSlave = getSameChannels(histAcc, 1)

        binDurationMaster = 23.616e-12
        binDurationSlave = 22.22e-12

        offsetMaster = 1559
        offsetSlave = 1591

        mergedImag = np.zeros((histoMaster.shape[0], len(bins)-1), dtype=np.float64)

        for i in range(histoMaster.shape[0]):
            singleHistoMaster = histoMaster[i]
            singleHistoSlave = histoSlave[i]
            valsNeeded = []
            for j in range(4096):
                timeMaster = float((j - offsetMaster)) * binDurationMaster
                timeSlave = float((j - offsetSlave)) * binDurationSlave

                #append for master
                valsNeeded.append([timeMaster] * singleHistoMaster[j])
                #append for slave
                valsNeeded.append([timeSlave] * singleHistoSlave[j])

            #ravel
            flat_list = [item for sublist in valsNeeded for item in sublist]
            ravelled = np.array(flat_list)
            histo = np.histogram(ravelled, bins)
            #add column to imag
            mergedImag[i] = histo[0]

        #got histos, create imag
        #jup

        # copy normalization one
        normedImag = mergedImag

        #transpose imag
        mergedImag = mergedImag.T


        #normalize
        imagBinnedMaster, imagBinnedSlave = self.correlator.getBinnedG1()
        totaloBinned = np.add(imagBinnedSlave[1], imagBinnedMaster[1])
        totaloBinned = totaloBinned / np.sum(totaloBinned)

        newNormalized = np.zeros_like(normedImag)

        for i in range(normedImag.shape[0]):
            newNormalized[i] = normedImag[i] / totaloBinned[i]

        finishedNormalied = newNormalized.T

        perBin = float(np.sum(mergedImag)) / float(mergedImag.shape[0] * mergedImag.shape[1])

        self.label_32.setText(str(np.round(perBin, 2)))

        #plot stuff
        self.PlotCanvasMerged.plot(mergedImag, finishedNormalied, None, None, "G2", "g2")

    def saveChunkedG1(self):
        self.Fileo = QFileDialog(self)
        self.Fileo.setAcceptMode(QFileDialog.AcceptOpen)
        self.Fileo.setFileMode(QFileDialog.AnyFile)

        self.Fileo.show()
        if self.Fileo.exec_():
            imagBinnedMaster, imagBinnedSlave = self.correlator.getBinnedG1()
            saveArray = np.array([imagBinnedMaster, imagBinnedSlave])
            np.save(self.Fileo.selectedFiles()[0], saveArray)
            QMessageBox.about(self, 'Success', "Saving of Chunked G1 successful")


    def refreshg2Histo(self):
        self.printHistogram("", True)

    def printHistogram(self, filename, onlyRefreshg2=False):

        def getSameChannels(histo, detector=0):
            len = histo.shape[1]
            out = np.zeros((len, histo.shape[3]), dtype=np.int32)
            for i in range(len):
                out[i] = histo[detector][i][i]
            return out

        def getCrossChannels(histo, startChannel, detector=0):
            len = histo.shape[1]
            out = np.zeros((len, histo.shape[3]), dtype=np.int32)
            for i in range(len):
                out[i] = histo[detector][startChannel][i]
            return out

        histoNow, histAcc = self.correlator.getHistos()


        if (onlyRefreshg2 == False):
            nowM, nowS, accM, accS =  self.correlator.getImages()
            self.PlotCanvasNormed.plot(nowM, nowS, accM, accS)
            self._totalG1Array = np.array([accM, accS])




            if (self.label_6.text() != "NotSet"):

                rawName = str(filename).split(".")[0]

                plt.imshow(nowM)
                plt.title("Master Shot")
                plt.savefig(self.label_6.text() + "/" + rawName + "Master.png",dpi=150)

                plt.imshow(nowS)
                plt.title("Slave Shot")
                plt.savefig(self.label_6.text() + "/" + rawName + "Slave.png", dpi=150)



            #calc new found double evs
            doubleEvs = np.sum(histoNow)

            self.threadPrinter("found " + str(doubleEvs) + " double Events in the file")
            self.label_19.setText(str(doubleEvs))

            #generate imags

            # test plotting
            rawVals = getSameChannels(histoNow, 0)
            # rebin second axis
            # how many rebins?
            rebinner = 64  # factorial of 2 (4, 8, 16, 32, ...)!
            rebinned = np.zeros((rawVals.shape[0], int(rawVals.shape[1] / rebinner)), dtype=np.int32)
            for i in range(rawVals.shape[0]):
                rebinned[i] = np.sum(rawVals[i].reshape(-1, rebinner), axis=1)
            histMasterNow = rebinned.T

            # test plotting
            rawVals = getSameChannels(histoNow, 1)
            # rebin second axis
            # how many rebins?
            rebinner = 64  # factorial of 2 (4, 8, 16, 32, ...)!
            rebinned = np.zeros((rawVals.shape[0], int(rawVals.shape[1] / rebinner)), dtype=np.int32)
            for i in range(rawVals.shape[0]):
                rebinned[i] = np.sum(rawVals[i].reshape(-1, rebinner), axis=1)
            histSlaveNow = rebinned.T

        # test plotting
        rawVals = getSameChannels(histAcc, 0)
        # rebin second axis
        # how many rebins?
        rebinner = 64  # factorial of 2 (4, 8, 16, 32, ...)!
        rebinned = np.zeros((rawVals.shape[0], int(rawVals.shape[1] / rebinner)), dtype=np.int32)
        for i in range(rawVals.shape[0]):
            rebinned[i] = np.sum(rawVals[i].reshape(-1, rebinner), axis=1)
        histMasterAcc = rebinned.T

        # test plotting
        rawVals = getSameChannels(histAcc, 1)
        # rebin second axis
        # how many rebins?
        rebinner = 64  # factorial of 2 (4, 8, 16, 32, ...)!
        rebinned = np.zeros((rawVals.shape[0], int(rawVals.shape[1] / rebinner)), dtype=np.int32)
        for i in range(rawVals.shape[0]):
            rebinned[i] = np.sum(rawVals[i].reshape(-1, rebinner), axis=1)
        histSlaveAcc = rebinned.T

        #plot
        if (onlyRefreshg2 == False):
            self.PlotCanvasG2.plot(histMasterNow, histSlaveNow, histMasterAcc, histSlaveAcc)

        # add plotting of binned G1 here!!
        imagBinnedMaster, imagBinnedSlave = self.correlator.getBinnedG1()

        normValMaster = np.sum(imagBinnedMaster)**2
        normValSlave = np.sum(imagBinnedSlave)**2

        normValHistM = np.sum(histMasterAcc)
        normValHistS = np.sum(histSlaveAcc)

        #do rebinning for g2
        rawVals = getSameChannels(histAcc, 0)
        # rebin second axis
        # how many rebins?
        rebinner = 64  # factorial of 2 (4, 8, 16, 32, ...)!
        rebinned = np.zeros((rawVals.shape[0], int(rawVals.shape[1] / rebinner)), dtype=np.float64)
        for i in range(rawVals.shape[0]):
            totaloMean = (imagBinnedMaster[1][i] ** 2) * 0.1
            rebinned[i] = (1. / (totaloMean /normValMaster)) * (np.sum(rawVals[i].reshape(-1, rebinner), axis=1)/normValHistM)
        normedMaster = rebinned.T

        # do rebinning for g2
        rawVals = getSameChannels(histAcc, 1)
        # rebin second axis
        # how many rebins?
        rebinner = 64  # factorial of 2 (4, 8, 16, 32, ...)!
        rebinned = np.zeros((rawVals.shape[0], int(rawVals.shape[1] / rebinner)), dtype=np.float64)
        for i in range(rawVals.shape[0]):
            totaloMean = (imagBinnedSlave[1][i]**2)*0.1
            rebinned[i] = (1. / (totaloMean/normValSlave)) * (np.sum(rawVals[i].reshape(-1, rebinner), axis=1)/normValHistS)
        normedSlave = rebinned.T

        #plot normlaized with zero only bin
        self.PlotCanvasg2.plot(normedMaster, normedSlave, imagBinnedMaster, imagBinnedSlave, "Master normed", "Slave normed", "G1 Master", "G1 Slave", float(self.lineEdit_2.text()), float(self.lineEdit.text()))
        self.PlotCanvasT0.plot(np.array([range(int(self.lineEdit_3.text())), normedMaster[24]]), np.array([range(int(self.lineEdit_3.text())), normedSlave[24]]), "Master Tau 0", "Slave Tau 0")

        #plot two spatial positions
        if (self.comboBox_2.currentText() == "Master"):
            self.PlotCanvasdx.plot(np.array([range(64), normedMaster.T[int(self.lineEdit_4.text())]]), np.array([range(64), normedMaster.T[int(self.lineEdit_5.text())]]),"Position One", "Position Two")
        if (self.comboBox_2.currentText() == "Slave"):
            self.PlotCanvasdx.plot(np.array([range(64), normedSlave.T[int(self.lineEdit_4.text())]]), np.array([range(64), normedSlave.T[int(self.lineEdit_5.text())]]),"Position One", "Position Two")

        if (onlyRefreshg2):
            return 0

        #update Info
        self.label_14.setText(str(np.sum(histMasterAcc)))
        self.label_16.setText(str(np.sum(histSlaveAcc)))
        self.label_15.setText(str(np.round(float(np.sum(histMasterAcc)/float(50*int(self.lineEdit_3.text()))), 2)))
        self.label_17.setText(str(np.round(float(np.sum(histSlaveAcc)/float(50*int(self.lineEdit_3.text()))), 2)))

        #QMessageBox.about(self, 'Not Done Yet', "Not Done Yet")
        return 0
        #self.PlotCanvasNormed.plot([self.correlator.getHistograms()[1][:-1]*1e-12*float(self.lineEdit_3.setText()), self.correlator.getHistograms()[0]])





app = QApplication(sys.argv)
app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
w = AppWindow()
w.show()
sys.exit(app.exec_())
