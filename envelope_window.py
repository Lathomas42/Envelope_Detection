from pyqtgraph.Qt import QtGui, QtCore  # (the example applies equally well to PySide)
import pyqtgraph as pg
import glob
import envelope_functions as ef
import numpy as np
import time

## Always start by initializing Qt (only once per application)
app = QtGui.QApplication([])

## Define a top-level widget to hold everything
w = QtGui.QWidget()

l_rms = None
l_rms2 = None
l_fdlp = None
l_fdlp2=None
l_tae = None
l_tae2 = None
audio_x = None
audio_y = None
freq = None
filt = None
curItem = None
fn_list = glob.glob("*.wav")

## Create some widgets to be placed inside
f_list = QtGui.QListWidget()
for fn in fn_list:
    f_list.addItem(fn)
btn_rms = QtGui.QPushButton('RMS Envelope')
btn_fdlp = QtGui.QPushButton('FDLP Envelope')
btn_tae = QtGui.QPushButton('TAE Envelope')
text = QtGui.QLineEdit('enter text')
p1 = pg.PlotWidget()
p2 = pg.PlotWidget()
p2.setYRange(0,1)
lr = pg.LinearRegionItem()
p1.addItem(lr)
## Create a grid layout to manage the widgets size and position
layout = QtGui.QGridLayout()
w.setLayout(layout)

layout.addWidget(f_list,0,0)
layout.addWidget(btn_rms, 1, 0)
label_rms = QtGui.QLabel("Frame Size (power of two): 512")
time_rms = QtGui.QLabel("Time to complete:")
spin_rms = pg.SpinBox(value=9, int=True, bounds=[0, 15],minStep=1,step=1)
layout.addWidget(label_rms,2,0)
layout.addWidget(time_rms,3,0)
layout.addWidget(spin_rms,4,0)
def val_rms(sb):
    label_rms.setText("Frame Size (power of two): {}".format(2**sb.value()))
spin_rms.sigValueChanged.connect(val_rms)
    
layout.addWidget(btn_fdlp,5,0)
label_fdlp = QtGui.QLabel("Order: 10")
time_fdlp = QtGui.QLabel("Time to complete:")
spin_fdlp = pg.SpinBox(value=10, int=True, bounds=[1,200],minStep=1,step=5)
layout.addWidget(label_fdlp,6,0)
layout.addWidget(time_fdlp,7,0)
layout.addWidget(spin_fdlp,8,0)
def val_fdlp(sb):
    label_fdlp.setText("Order: {}".format(sb.value()))
spin_fdlp.sigValueChanged.connect(val_fdlp)

layout.addWidget(btn_tae,9,0)
label_tae = QtGui.QLabel("Frame Size (power of two): 512")
time_tae = QtGui.QLabel("Time to complete:")
spin_tae = pg.SpinBox(value=9, int=True, bounds=[0, 15],minStep=1,step=1)
layout.addWidget(label_tae,10,0)
layout.addWidget(time_tae,11,0)
layout.addWidget(spin_tae,12,0)
def val_tae(sb):
    label_tae.setText("Frame Size (power of two): {}".format(2**sb.value()))
spin_tae.sigValueChanged.connect(val_tae)    

btn_filter = QtGui.QPushButton('Global Frequency Selector')
label_filter = QtGui.QLabel("Select Frequency to Capture")
spin_filter = pg.SpinBox(value = 0,int=True,bounds=[0,20000],minStep=1,step=10)
layout.addWidget(btn_filter,13,0)
layout.addWidget(label_filter,14,0)
layout.addWidget(spin_filter,15,0)
def val_filter(sb):
    global filt
    val = sb.value()
    if val == 0:
        label_filter.setText("Select Frequency to Capture")
        filt = None
    else:
        label_filter.setText("Frequency Selected: {}".format(sb.value()))
        filt = val
    if curItem is not None:
        show_raw(curItem)
spin_filter.sigValueChanged.connect(val_filter)

## Add widgets to the layout in their proper positions
layout.addWidget(p1, 0, 1, 6, 1)  # plot goes on right side, spanning 3 rows
layout.addWidget(p2, 7, 1, 10, 1)
## Display the widget as a new window
w.show()


def show_raw(item):
    global audio_x, audio_y, freq, curItem, filt
    curItem = item
    if l_rms is not None:
        p1.removeItem(l_rms)
        p2.removeItem(l_rms2)
    if l_fdlp is not None:
        p1.removeItem(l_fdlp)
        p2.removeItem(l_fdlp2)
    if l_tae is not None:
        p1.removeItem(l_tae)
        p2.removeItem(l_tae2)
    fn = item.text()
    w = ef.load_audio(fn)
    freq = w[2]
    p1.clear()
    p2.clear()
    x_r = np.arange(len(w[0]))/float(w[2])
    audio_x = x_r
    audio_y = w[0]/max(abs(w[0]))
    if filt is not None:
        ind = int(filt * len(audio_y)/freq)
        if ind > 1024:
            filter = np.append(np.zeros(ind-1024),np.hamming(2048),np.zeros(len(audio_y)-ind-1024))
        else:
            filter = np.append(np.hamming(2*ind),np.zeros(len(audio_y)-2*ind))
        audio_y = np.real(np.fft.ifft(np.fft.fft(audio_y)*filter))
        audio_y = audio_y/max(abs(audio_y))
    p1.plot(audio_x,audio_y,pen=(1,4))
    lr.setBounds([x_r[0],x_r[-1]])
    lr.setRegion([x_r[0],x_r[-1]])
    p1.addItem(lr)
    p2.plot(audio_x,np.abs(audio_y),pen=(1,4))
    
f_list.itemDoubleClicked.connect(show_raw)

def updatePlot():
    p2.setXRange(*lr.getRegion(), padding=0)

def updateRegion():
    lr.setRegion(p2.getViewBox().viewRange()[0])

lr.sigRegionChangeFinished.connect(updatePlot)
p2.sigXRangeChanged.connect(updateRegion)
updatePlot()


def rms_show():
    global l_rms, l_rms2, audio_x, audio_y, freq
    if l_rms is not None:
        p1.removeItem(l_rms)
        p2.removeItem(l_rms2)
    w_size = 2**spin_rms.value()
    n_windows = int(len(audio_y)/w_size)
    x_vals = np.arange(n_windows)*w_size*(1./freq)
    t_init = time.time()
    t_i = time.time()
    r = ef.rms_total(audio_y,w_size)
    t_f = time.time()
    l_rms = p1.plot(x_vals,r/max(r),pen=(2,4))
    l_rms2 = p2.plot(x_vals,r/max(r),pen=(2,4))
    time_rms.setText("Time to complete: {} ms".format(round((t_f-t_i)*1000.,2)))
btn_rms.clicked.connect(rms_show)

def fdlp_show():
    global l_fdlp,l_fdlp2,audio_x,audio_y,freq
    if l_fdlp is not None:
        p1.removeItem(l_fdlp)
        p2.removeItem(l_fdlp2)
    order = spin_fdlp.value()
    t_i = time.time()
    x_v, y_v = ef.FDLP(audio_y,order)
    x_v = 1.0*x_v*1/freq
    t_f = time.time()
    l_fdlp = p1.plot(x_v,y_v,pen=(3,4))
    l_fdlp2 = p2.plot(x_v,y_v,pen=(3,4))
    time_fdlp.setText("Time to complete: {} ms".format(round((t_f-t_i)*1000.,2)))
btn_fdlp.clicked.connect(fdlp_show)

def tae_show():
    global l_tae, l_tae2, audio_x, audio_y, freq
    if l_tae is not None:
        p1.removeItem(l_tae)
        p2.removeItem(l_tae2)
    w_size = 2**spin_tae.value()
    t_i = time.time()
    y_v = ef.tae(audio_y,w_size)
    t_f = time.time()
    y_v = y_v/max(y_v)
    x_v = audio_x
    l_tae = p1.plot(x_v,y_v,pen=(4,4))
    l_tae2 = p2.plot(x_v,y_v,pen=(4,4))
    time_tae.setText("Time to complete: {} ms".format(round((t_f-t_i)*1000.,2)))
btn_tae.clicked.connect(tae_show)   










## Start the Qt event loop
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
