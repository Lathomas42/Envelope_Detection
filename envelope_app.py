import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
import glob 
import envelope_functions as efunct
import scipy.io.wavfile

env1 = None
env2 = None
fn = 'aaa.wav'
fig, axarr = plt.subplots(2)
a_1,a_2,freq = efunct.load_audio(fn)
plt.subplots_adjust(left=0.3)

axcolor = 'lightgoldenrodyellow'
rax = plt.axes([0.05, 0.7, 0.15, 0.15], axisbg=axcolor)
rax.set_title('File Selector')
fns = glob.glob('*.wav')
radio = RadioButtons(rax, fns)
def file_selector(label):
    global a_1,a_2,fn,freq
    fn = label
    a_1,a_2,freq = efunct.load_audio(fn)
    show_audio(a_1,a_2,freq)
    plt.draw()
radio.on_clicked(file_selector)

rax = plt.axes([0.05, 0.5, 0.15, 0.15], axisbg=axcolor)
rax.set_title('RMS envelope')
radio2 = RadioButtons(rax, ('128 window', '256 window', '512 window'))
def rms_env(label):
    global a_1, a_2
    window_dict = {'128 window':128,'256 window':256,'512 window':512}
    w_size = window_dict[label]
    n_windows = int(len(a_1)/w_size)
    x_vals = np.arange(n_windows)*w_size*(1./freq)
    r_1 = efunct.rms_total(a_1,window_dict[label])
    r_2 = efunct.rms_total(a_2,window_dict[label])
    show_envelope(x_vals,r_1,x_vals,r_2)
    plt.draw()
radio2.on_clicked(rms_env)

rax = plt.axes([0.05, 0.3, 0.15, 0.15], axisbg=axcolor)
rax.set_title("Hilbert Envelope")
radio3 = RadioButtons(rax, ('-', '--', '-.', 'steps', ':'))
def hilbert_env(label):
    global a_1, a_2, freq
    h_1 = efunct.hilbert(a_1)
    h_2 = efunct.hilbert(a_2)
    time = np.arange(len(a_2))*1./freq
    show_envelope(time,abs(h_1),time,abs(h_2))
    plt.draw()
radio3.on_clicked(hilbert_env)


def show_audio(aud1,aud2, freq):
    global axarr,env1, env2
    axarr[0].clear()
    axarr[1].clear()
    time = np.arange(len(aud2))*1./freq
    axarr[0].plot(time,aud1, color='red')
    axarr[1].plot(time,aud2,color='red')
    env1 = None
    env2 = None

def show_envelope(x1,y1,x2,y2):
    global env1, env2
    if env1 is None:
        env1, = axarr[0].plot(x1,y1,color='blue')
    else:
        env1.set_data(x1,y1)
    if env2 is None:
        env2, = axarr[1].plot(x2,y2,color='blue')
    else:
        env2.set_data(x2,y2)
        
show_audio(a_1,a_2,freq)

plt.show()