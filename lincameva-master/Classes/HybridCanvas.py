from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QDialog, QApplication,  QMainWindow, QFileDialog, QMessageBox, QSizePolicy
import random

import matplotlib.pyplot as plt

from .DraggableColorBar import DraggableColorbar
from .mynormalize import MyNormalize

import numpy as np

class HybridCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#31363B')
        self.axes = fig.add_subplot(121)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plot(self, imagOne=None, imagTwo=None, dataThree=None, dataFour=None, TitleOne=None, TitleTwo=None, TitleThree=None, TitleFour=None, vMin=0, vMax=3500):

        def customTicksM(value, tick_number):
            return np.round(float(value - 24) * 1.511424e-9, 9)

        def customTicksS(value, tick_number):
            return np.round(float(value - 24) * 1.422464e-9, 9)

        self.figure.clf()
        if imagOne is not None:
            ax = self.figure.add_subplot(221)
            ax.set_facecolor('#31363B')
            if TitleOne is None:
                ax.set_title("Master now")
            else:
                ax.set_title(TitleOne)
            imagOneHandle = ax.imshow(imagOne, vmin=vMin, vmax=vMax)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(customTicksM))
            self.figure.colorbar(imagOneHandle)

        if imagTwo is not None:
            axTwo = self.figure.add_subplot(222)
            axTwo.set_facecolor('#31363B')
            if TitleTwo is None:
                axTwo.set_title("Slave now")
            else:
                axTwo.set_title(TitleTwo)
            imagTwoHandle = axTwo.imshow(imagTwo, vmin=vMin, vmax=vMax)
            axTwo.yaxis.set_major_formatter(plt.FuncFormatter(customTicksS))
            self.figure.colorbar(imagTwoHandle)


        if dataThree is not None:
            axThree = self.figure.add_subplot(223)
            axThree.set_facecolor('#31363B')
            if TitleThree is None:
                axThree.set_title("Master Acc")
            else:
                axThree.set_title(TitleThree)
            dataThreeHandle = axThree.plot(dataThree[0], dataThree[1])

        if dataFour is not None:
            axFour = self.figure.add_subplot(224)
            axFour.set_facecolor('#31363B')
            if TitleFour is None:
                axFour.set_title("Slave Acc")
            else:
                axFour.set_title(TitleFour)
            dataFourHandle = axFour.plot(dataFour[0], dataFour[1])

        self.draw()