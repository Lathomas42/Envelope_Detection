import scipy.io.wavfile
import numpy
import time
import matplotlib.pyplot as plt

def load_audio(fn='aaa.wav'):
	w_file = scipy.io.wavfile.read(fn)
	freq = w_file[0]
	audio = w_file[1]
	audio_L = audio[:,0]
	audio_R = audio[:,1]
	#pad with zeros to make a power of two length
	pow2 = numpy.log2(len(audio_L))
	if round(pow2) != numpy.log2(pow2):
		new_audio_L = numpy.zeros(pow(2,numpy.ceil(pow2)))
		new_audio_R = new_audio_L.copy()
		new_audio_L[0:len(audio_L)] = audio_L
		new_audio_R[0:len(audio_R)] = audio_R
		audio_L = new_audio_L
		audio_R = new_audio_R
		pow2 = numpy.log2(len(audio_L))
	return audio_L, audio_R, pow2

def rms(fn='aaa.wav', window_size=256):
	"""
	uses RMS energy method to define envelope for audio file in fn
	"""
	#moving window
	window = numpy.zeros(window_size)


	audio_L, audio_R, pow2 = load_audio(fn)
	st = time.time()
	#continue filling floating window until done
	n_windows = int(pow(2,pow2-numpy.log2(window_size)))
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
	plt.plot(numpy.arange(n_windows)*n_windows-n_windows/2.,rms_R)
	plt.subplot(212)
	plt.plot(numpy.arange(len(audio_L)),abs(audio_L))
	plt.plot(numpy.arange(n_windows)*n_windows-n_windows/2,rms_L)
	plt.show()

def hilbert(window, freq):
	"""
	returns the hilbert transform of the window of audio samples at a given 
	frequency
	hilbert transform is defined as the convolution of x(t) and 1/(pi*t)
	"""
	return False

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
	

def DCT(x):
	"""
	Discrete Cosine Transform algorithm
	"""
	a_vals = numpy.ones(x.shape[0])
	a_vals[1:] = numpy.sqrt(2)
	ft = FFT(x)
	th = numpy.arctan(ft.imag/ft.real)
	ft = numpy.absolute(ft)
	n = x.shape[0]
	return a_vals*ft*numpy.cos(th-numpy.pi*numpy.arange(n)/(2*n))

