# -*- coding: utf-8 -*-
"""
This example demonstrates many of the 2D plotting capabilities
in pyqtgraph. All of the plots may be panned/scaled by dragging with 
the left/right mouse buttons. Right click on any plot to show a context menu.
"""

from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg

#QtGui.QApplication.setGraphicsSystem('raster')
app = QtGui.QApplication([])
#mw = QtGui.QMainWindow()
#mw.resize(800,800)



win = pg.GraphicsWindow(title="Basic plotting examples")
win.resize(1000,600)
win.setWindowTitle('pyqtgraph example: Plotting')

# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)




layout = QtGui.QGridLayout()

spins = [
    ("Floating-point spin box, min=0, no maximum.", pg.SpinBox(value=5.0, bounds=[0, None])),
    ("Integer spin box, dec stepping<br>(1-9, 10-90, 100-900, etc)", pg.SpinBox(value=10, int=True, dec=True, minStep=1, step=1)),
    ("Float with SI-prefixed units<br>(n, u, m, k, M, etc)", pg.SpinBox(value=0.9, suffix='V', siPrefix=True)),
    ("Float with SI-prefixed units,<br>dec step=0.1, minStep=0.1", pg.SpinBox(value=1.0, suffix='V', siPrefix=True, dec=True, step=0.1, minStep=0.1)),
    ("Float with SI-prefixed units,<br>dec step=0.5, minStep=0.01", pg.SpinBox(value=1.0, suffix='V', siPrefix=True, dec=True, step=0.5, minStep=0.01)),
    ("Float with SI-prefixed units,<br>dec step=1.0, minStep=0.001", pg.SpinBox(value=1.0, suffix='V', siPrefix=True, dec=True, step=1.0, minStep=0.001)),
]
changedLabel = QtGui.QLabel()
font = changedLabel.font()
font.setBold(True)
font.setPointSize(12)
changedLabel.setFont(font)
labels=[]
def valueChanged(sb):
    changedLabel.setText("Final value: %s" % str(sb.value()))

def valueChanging(sb, value):
    changingLabel.setText("Value changing: %s" % str(sb.value()))

    
for text, spin in spins:
    label = QtGui.QLabel(text)
    labels.append(label)
    layout.addWidget(label)
    layout.addWidget(spin)
    spin.sigValueChanged.connect(valueChanged)
    spin.sigValueChanging.connect(valueChanging)




x2 = np.linspace(-100, 100, 1000)
data2 = np.sin(x2) / x2
p8 = win.addPlot(title="Region Selection")
p8.plot(data2, pen=(255,255,255,200))
lr = pg.LinearRegionItem([400,700])
lr.setZValue(-10)
p8.addItem(lr)

p9 = win.addPlot(title="Zoom on selected region")
p9.plot(data2)
def updatePlot():
    p9.setXRange(*lr.getRegion(), padding=0)
def updateRegion():
    lr.setRegion(p9.getViewBox().viewRange()[0])
lr.sigRegionChanged.connect(updatePlot)
p9.sigXRangeChanged.connect(updateRegion)
updatePlot()

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
