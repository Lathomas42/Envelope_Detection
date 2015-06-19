import scipy.io.wavfile
import scipy.fftpack
import numpy
import time
import matplotlib.pyplot as plt

    

def make_pow_2(x):
	"""
	make an array a power of 2 above its current size (if not a power of 2)
	"""
	pow2 = numpy.log2(x.shape[0])
	if round(pow2) != pow2:
		# if not a whole number power of 2
		n_x = numpy.zeros(pow(2,numpy.ceil(pow2)))
		n_x[0:x.shape[0]] = x
		x = n_x
		pow2 = numpy.log2(x.shape[0])
	return x,pow2

def load_audio(fn='aaa.wav'):
    w_file = scipy.io.wavfile.read(fn)
    freq = w_file[0]
    audio = w_file[1]
    if(audio.ndim > 1 ):
        audio_L = audio[:,0]
        audio_R = audio[:,1]
        #pad with zeros to make a power of two length
    else:
        audio_L = audio
        audio_R = audio
    audio_L, pow2 = make_pow_2(audio_L)
    audio_R, pow2 = make_pow_2(audio_R)
    return audio_L, audio_R, freq

def rms(x,window_size):
	return numpy.sqrt(sum(x**2)/window_size)

def rms_total(x, window_size=256):
	n_windows = int(pow(2,numpy.log2(len(x))-numpy.log2(window_size)))
	rms_tot = numpy.zeros(n_windows)
	for i in range(n_windows):
		w = x[i*window_size:(i+1)*window_size]
		rms_tot[i] = rms(w,window_size)
	return rms_tot

def rms_envelope(fn='aaa.wav', window_size=256):
	"""
	uses RMS energy method to define envelope for audio file in fn
	"""
	#moving window
	window = numpy.zeros(window_size)


	audio_L, audio_R, freq = load_audio(fn)
	st = time.time()
	#continue filling floating window until done
	n_windows = int(len(audio_L)/window_size)
	rms_L = numpy.zeros(n_windows)
	rms_R = rms_L.copy()
	for i in range(n_windows):
		window_L = audio_L[i*window_size:(i+1)*window_size]
		window_R = audio_R[i*window_size:(i+1)*window_size]
		rms_L[i] = numpy.sqrt(sum(window_L**2)/window_size)
		rms_R[i] = numpy.sqrt(sum(window_R**2)/window_size)
	print time.time()-st
	plt.figure()
	plt.subplot(211)
	plt.plot(numpy.arange(len(audio_R)),abs(audio_R))
	plt.plot(numpy.arange(n_windows)*window_size,rms_R)
	plt.subplot(212)
	plt.plot(numpy.arange(len(audio_L)),abs(audio_L))
	plt.plot(numpy.arange(n_windows)*window_size-window_size/2,rms_L)
	plt.show()
#--------------------------
def hilbert(x):
	"""
	returns the hilbert transform of the window of audio samples at a given 
	frequency
	hilbert transform is defined as the convolution of x(t) and 1/(pi*t)
	"""
	n = x.shape[0]
	f = numpy.fft.fft(x)
	h = numpy.zeros(n)
	if n%2 == 0:
		#even
		h[0] = h[n//2]=1
		h[1:n//2]=2
	else:
		h[0] = 1
		h[1:(n+1)//2] = 2
	x = numpy.fft.ifft(f*h)
	return x

def hilbert_envelope(fn="aaa.wav"):
	"""
	plots the hilbert envelope
	"""
	L,R,f = load_audio(fn)
	h_L = hilbert(L)
	h_R = hilbert(R)
	plt.figure()
	plt.plot(L)
	plt.plot(abs(h_L))
	plt.figure()
	plt.plot(R)
	plt.plot(abs(h_R))
#----------------------------
def FFT(x):
	"""
	recursive implimentation of FFT
	"""
	x = numpy.asarray(x, dtype=float)
	N = x.shape[0]
	if N<32:
		return DFT(x)
	else:	
		X_even = FFT(x[::2])
		X_odd = FFT(x[1::2])
		factor = numpy.exp(-2j * numpy.pi * numpy.arange(N)/N)
		return numpy.concatenate([X_even + factor[:N / 2]*X_odd,
					  X_even + factor[N / 2:] * X_odd])

def DFT(x):
	"""
	discrete fourier transform
	"""
	x = numpy.asarray(x,dtype=float)
	N = x.shape[0]
	n = numpy.arange(N)
	k = n.reshape((N,1))
	M = numpy.exp(-2j * numpy.pi * k * n/N)
	return numpy.dot(M,x)
	
#--------------------------------
# def DCT(x):
	# """
	# Discrete Cosine Transform algorithm
	# """
	# a_vals = numpy.ones(x.shape[0])
	# a_vals[1:] = numpy.sqrt(2)
	# ft = numpy.fft.fft(x)[0:x.shape[0]]
	# th = numpy.arctan(ft.imag/ft.real)
	# ft = numpy.absolute(ft)
	# n = x.shape[0]
	# return a_vals*ft*numpy.cos(th-numpy.pi*numpy.arange(n)/(2.*n))
#----------------------------------------------
def _tae(audio,threshold=None, order=20,power=False):
    '''
    Calculate the True Amplitude Envelope of the given audio signal,
    with the given order.
    '''
    # extra zero padding to nearest power of 2
    if not len(audio) % 2 == 0:
        next_power_of_2 = 2 ** numpy.ceil(numpy.log2(len(audio)))
        audio = numpy.hstack((audio, numpy.zeros(next_power_of_2 - len(audio))))

    N = len(audio)
    fwr = numpy.abs(audio)
    if power:
        fwr[numpy.nonzero(fwr)] = numpy.log(fwr[numpy.nonzero(fwr)])
    if threshold is None:
    	threshold = 0.025 * numpy.max(fwr)
    go = True
    while go:
        ceps = numpy.fft.ifft(fwr)
        ceps[order:N - order] = 0.0
        env = numpy.fft.fft(ceps).real
        fwr = numpy.maximum(fwr, env)
        go = ((fwr - env) > threshold).any()
    return env

def tae(audio, frame_size=512, hop_size=None, order=None,power=False):
    '''
    Streaming version of the True Amplitude Envelope.
    '''
    env = numpy.zeros(len(audio))
    # TODO: should we calculate an optimal order here?
    #       this value seems to work well in practice
    if hop_size is None:
        hop_size = frame_size/2
    if order is None:
        order = 6
    p = 0
    t = .025*numpy.max(numpy.abs(audio))
    if power:
        t = .025 * numpy.log(numpy.max(numpy.abs(audio)))
    while p <= len(audio) - frame_size:
        frame = audio[p:p + frame_size]
        env[p:p + frame_size] += _tae(frame,t, order,power) * numpy.hanning(len(frame))
        p += hop_size
    return env


def toeplitz(arr):
    """
    creates a toeplitz matrix 2N-1 x 2N-1 where N is the size of arr
    """
    toep = numpy.zeros((len(arr),len(arr)))
    for i in numpy.arange(len(arr)):
        toep[:,i][i:] = arr[:len(arr)-i]
        toep[i,:][i:] = arr[:len(arr)-i]
    return toep

    
def autolpc(x,order=10):
    """
    computes LPC coefficients with numpys built in linear equation solver
    """
    L= len(x)
    r = numpy.zeros(order+1)
    for i in range(order+1):
        r[i] = numpy.dot(x[0:L-i],x[i:L])
    R = toeplitz(r[:order])
    a = numpy.linalg.solve(R,r[1:])
    A = numpy.append([1],-1.*a)
    G = numpy.sqrt(numpy.sum(A*r))
    return A, G, r, a
    
def FDLP(x,order=10, filter=None):
    """
    uses LPC to extract the original envelope from the DCT
    """
    d = scipy.fftpack.dct(x,norm='ortho')
    if filter is not None:
        d = d * filter 
    A,G,r,a = autolpc(d,order)
    H = G/numpy.fft.fft(A,len(d))
    half = H[0:len(H)/2]
    #half = numpy.sqrt(half)
    return numpy.arange(0,len(d),2),abs(half)/max(abs(half))
    #plt.plot(numpy.arange(0,len(d),2),abs(half)/max(abs(half)))

def FDLP_windowed(x,w_size,order=10,filter=None):
    n_windows = len(x)/w_size
    # TODO
    
def DCT(x):
    """ how DCT is calculated, scipy.fftpack.dft is faster though"""
    N = len(x)
    X = numpy.zeros(N)
    for i in numpy.arange(N):
        sum = 0.
        if i == 0:
            s = numpy.sqrt(.5)
        else:
            s = 1.
        for n in numpy.arange(N):
            sum += s * x[n] * numpy.cos(numpy.pi*(n+.5)*i/N)
        X[i] = sum * numpy.sqrt(2./N)
    return X