from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QDialog, QApplication,  QMainWindow, QFileDialog, QMessageBox, QSizePolicy
import random

import matplotlib.pyplot as plt

from .DraggableColorBar import DraggableColorbar
from .mynormalize import MyNormalize

import numpy as np

class DoublePlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#31363B')
        #self.axes = fig.add_subplot(121)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plot(self, dataOne=None, dataTwo=None, TitleOne=None, TitleTwo=None):

        self.figure.clf()

        if dataOne is not None:
            axThree = self.figure.add_subplot(211)
            axThree.set_facecolor('#31363B')
            if TitleOne is None:
                axThree.set_title("Master Acc")
            else:
                axThree.set_title(TitleOne)
            dataThreeHandle = axThree.plot(dataOne[0], dataOne[1])

        if dataTwo is not None:
            axFour = self.figure.add_subplot(212)
            axFour.set_facecolor('#31363B')
            if TitleTwo is None:
                axFour.set_title("Slave Acc")
            else:
                axFour.set_title(TitleTwo)
            dataFourHandle = axFour.plot(dataTwo[0], dataTwo[1])

        self.draw()