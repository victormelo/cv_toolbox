#!/usr/bin/env python

from PyQt4 import QtGui, QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from PIL import Image
from ..binary.kmeans import KMeans
from ..core.util import gradient

class PlotCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.axes.get_xaxis().set_visible(False)
        self.axes.get_yaxis().set_visible(False)
        # We want the axes cleared every time plot() is called
        self.fig.tight_layout(pad=0)
        self.axes.hold(False)

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class ImagePlotCanvas(PlotCanvas):
    """Simple canvas with a sine plot."""

    def __init__(self, *args, **kwargs):
        PlotCanvas.__init__(self, *args, **kwargs)

    def set_image(self, image):
        self.image = image
        self.update_figure()

    def update_figure(self):
        self.axes.imshow(self.image, cmap='gray')
        # self.fig.tight_layout(pad=0)
        self.draw()


class ImageViewer(QtGui.QMainWindow):
    def __init__(self):
        super(ImageViewer, self).__init__()

        self.printer = QtGui.QPrinter()

        self.main_widget = QtGui.QWidget(self)

        self.createActions()
        self.createMenus()

        self.main_widget = QtGui.QWidget(self)

        l = QtGui.QVBoxLayout(self.main_widget)
        self.imageCanvas = ImagePlotCanvas(
            self.main_widget, width=2, height=2, dpi=100)
        l.addWidget(self.imageCanvas)

        def onclick(event):
            print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
                  (event.button, event.x, event.y, event.xdata, event.ydata))

        cid = self.imageCanvas.mpl_connect('button_press_event', onclick)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

    def open(self):
        fileName = QtGui.QFileDialog.getOpenFileName(self, "Open File",
                                                     QtCore.QDir.currentPath())
        if fileName:
            path = str(fileName)
            self.img = Image.open(path).convert('L')
        # print self.img
        # exit()
        self.imageCanvas.set_image(self.img)

    def print_(self):
        dialog = QtGui.QPrintDialog(self.printer, self)
        if dialog.exec_():
            painter = QtGui.QPainter(self.printer)
            rect = painter.viewport()
            size = self.imageLabel.pixmap().size()
            size.scale(rect.size(), QtCore.Qt.KeepAspectRatio)
            painter.setViewport(
                rect.x(), rect.y(), size.width(), size.height())
            painter.setWindow(self.imageLabel.pixmap().rect())
            painter.drawPixmap(0, 0, self.imageLabel.pixmap())

    def zoomIn(self):
        self.scaleImage(1.25)

    def zoomOut(self):
        self.scaleImage(0.8)

    def normalSize(self):
        self.imageLabel.adjustSize()
        self.scaleFactor = 1.0

    def fitToWindow(self):
        fitToWindow = self.fitToWindowAct.isChecked()
        self.scrollArea.setWidgetResizable(fitToWindow)
        if not fitToWindow:
            self.normalSize()

        self.updateActions()

    def gradient(self):
        self.img, _ = gradient(self.img)
        self.imageCanvas.set_image(self.img)

    def about(self):
        QtGui.QMessageBox.about(self, "About Image Viewer",
                                "<p>The <b>Image Viewer</b> example shows how to combine "
                                "QLabel and QScrollArea to display an image. QLabel is "
                                "typically used for displaying text, but it can also display "
                                "an image. QScrollArea provides a scrolling view around "
                                "another widget. If the child widget exceeds the size of the "
                                "frame, QScrollArea automatically provides scroll bars.</p>"
                                "<p>The example demonstrates how QLabel's ability to scale "
                                "its contents (QLabel.scaledContents), and QScrollArea's "
                                "ability to automatically resize its contents "
                                "(QScrollArea.widgetResizable), can be used to implement "
                                "zooming and scaling features.</p>"
                                "<p>In addition the example shows how to use QPainter to "
                                "print an image.</p>")

    def createActions(self):
        self.openAct = QtGui.QAction("&Open...", self, shortcut="Ctrl+O",
                                     triggered=self.open)

        self.printAct = QtGui.QAction("&Print...", self, shortcut="Ctrl+P",
                                      enabled=False, triggered=self.print_)

        self.exitAct = QtGui.QAction("E&xit", self, shortcut="Ctrl+Q",
                                     triggered=self.close)

        self.zoomInAct = QtGui.QAction("Zoom &In (25%)", self,
                                       shortcut="Ctrl++", enabled=False, triggered=self.zoomIn)

        self.zoomOutAct = QtGui.QAction("Zoom &Out (25%)", self,
                                        shortcut="Ctrl+-", enabled=False, triggered=self.zoomOut)

        self.normalSizeAct = QtGui.QAction("&Normal Size", self,
                                           shortcut="Ctrl+S", enabled=False, triggered=self.normalSize)

        self.fitToWindowAct = QtGui.QAction("&Fit to Window", self,
                                            enabled=False, checkable=True, shortcut="Ctrl+F",
                                            triggered=self.fitToWindow)

        self.gradientAct = QtGui.QAction(
            "&Gradient", self, triggered=self.gradient)

        self.aboutAct = QtGui.QAction("&About", self, triggered=self.about)

        self.aboutQtAct = QtGui.QAction("About &Qt", self,
                                        triggered=QtGui.qApp.aboutQt)

    def createMenus(self):
        self.fileMenu = QtGui.QMenu("&File", self)
        self.fileMenu.addAction(self.openAct)
        self.fileMenu.addAction(self.printAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAct)

        self.viewMenu = QtGui.QMenu("&View", self)
        self.viewMenu.addAction(self.zoomInAct)
        self.viewMenu.addAction(self.zoomOutAct)
        self.viewMenu.addAction(self.normalSizeAct)
        self.viewMenu.addSeparator()
        self.viewMenu.addAction(self.fitToWindowAct)

        self.preproccessMenu = QtGui.QMenu("&Preproccess", self)
        self.preproccessMenu.addAction(self.gradientAct)

        self.helpMenu = QtGui.QMenu("&Help", self)
        self.helpMenu.addAction(self.aboutAct)
        self.helpMenu.addAction(self.aboutQtAct)

        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.viewMenu)
        self.menuBar().addMenu(self.preproccessMenu)
        self.menuBar().addMenu(self.helpMenu)

    def updateActions(self):
        self.zoomInAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.zoomOutAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.normalSizeAct.setEnabled(not self.fitToWindowAct.isChecked())

    def scaleImage(self, factor):
        self.scaleFactor *= factor
        self.imageLabel.resize(
            self.scaleFactor * self.imageLabel.pixmap().size())

        self.adjustScrollBar(self.scrollArea.horizontalScrollBar(), factor)
        self.adjustScrollBar(self.scrollArea.verticalScrollBar(), factor)

        self.zoomInAct.setEnabled(self.scaleFactor < 3.0)
        self.zoomOutAct.setEnabled(self.scaleFactor > 0.333)

    def adjustScrollBar(self, scrollBar, factor):
        scrollBar.setValue(int(factor * scrollBar.value()
                               + ((factor - 1) * scrollBar.pageStep() / 2)))

def main():
    import sys

    app = QtGui.QApplication(sys.argv)
    imageViewer = ImageViewer()
    imageViewer.show()
    sys.exit(app.exec_())


if __name__ == '__main__':

    main()
