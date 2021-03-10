from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QDialog, QApplication,  QMainWindow, QFileDialog, QMessageBox, QSizePolicy
import random

import numpy as np

class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#31363B')
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plot(self, data=None):
        self.figure.clf()
        if data is None:
            data = np.zeros((2, 25), dtype=np.float64)
            data[0] = range(25)
            data[1] = [random.random() for i in range(25)]
        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#31363B')
        ax.plot(data[0], data[1],'o-')
        ax.set_xlabel('time')
        ax.set_ylabel(r'$G^{(2)}$')
        self.draw()