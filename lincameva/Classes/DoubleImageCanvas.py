from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QDialog, QApplication,  QMainWindow, QFileDialog, QMessageBox, QSizePolicy
import random

import numpy as np

class DoubleImageCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#31363B')
        self.axes = fig.add_subplot(121)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plot(self, imagOne=None, imagTwo=None, imagThree=None, imagFour=None, TitleOne=None, TitleTwo=None, TitleThree=None, TitleFour=None, max=None, min=None):
        self.figure.clf()
        if imagOne is not None:
            ax = self.figure.add_subplot(221)
            ax.set_facecolor('#31363B')
            if TitleOne is None:
                ax.set_title("Master now")
            else:
                ax.set_title(TitleOne)
            imagOneHandle = ax.imshow(imagOne, vmin=min, vmax=max)
            self.figure.colorbar(imagOneHandle)

        if imagTwo is not None:
            axTwo = self.figure.add_subplot(222)
            axTwo.set_facecolor('#31363B')
            axTwo.imshow(imagTwo)
            if TitleTwo is None:
                axTwo.set_title("Slave now")
            else:
                axTwo.set_title(TitleTwo)
            imagTwoHandle = axTwo.imshow(imagTwo, vmin=min, vmax=max)
            self.figure.colorbar(imagTwoHandle)

        if imagThree is not None:
            axThree = self.figure.add_subplot(223)
            axThree.set_facecolor('#31363B')
            if TitleThree is None:
                axThree.set_title("Master Acc")
            else:
                axThree.set_title(TitleThree)
            imagThreeHandle = axThree.imshow(imagThree, vmin=min, vmax=max)
            self.figure.colorbar(imagThreeHandle)

        if imagFour is not None:
            axFour = self.figure.add_subplot(224)
            axFour.set_facecolor('#31363B')
            if TitleFour is None:
                axFour.set_title("Slave Acc")
            else:
                axFour.set_title(TitleFour)
            imagFourHandle = axFour.imshow(imagFour, vmin=min, vmax=max)
            self.figure.colorbar(imagFourHandle)

        self.draw()