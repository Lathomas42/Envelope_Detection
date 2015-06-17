import scipy.io.wavfile
import numpy
import time
import matplotlib.pyplot as plt

def pitch_selector(pitch, ft):
    return False
    

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
	audio_L = audio[:,0]
	audio_R = audio[:,1]
	#pad with zeros to make a power of two length
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
def DCT(x):
	"""
	Discrete Cosine Transform algorithm
	"""
	a_vals = numpy.ones(x.shape[0])
	a_vals[1:] = numpy.sqrt(2)
	ft = numpy.fft.fft(x)[0:x.shape[0]]
	th = numpy.arctan(ft.imag/ft.real)
	ft = numpy.absolute(ft)
	n = x.shape[0]
	return a_vals*ft*numpy.cos(th-numpy.pi*numpy.arange(n)/(2.*n))
#----------------------------------------------
def _tae(audio,threshold=None, order=20):
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

def tae(audio, frame_size=512, hop_size=256, order=None):
    '''
    Streaming version of the True Amplitude Envelope.
    '''
    env = numpy.zeros(len(audio))
    # TODO: should we calculate an optimal order here?
    #       this value seems to work well in practice
    if order is None:
        order = 6
    p = 0
    t = .025*numpy.max(numpy.abs(audio))
    while p <= len(audio) - frame_size:
        frame = audio[p:p + frame_size]
        env[p:p + frame_size] += _tae(frame, order) * numpy.hanning(len(frame))
        p += hop_size

    return env

def cepstral(x, n_c = 100):
    """
    Cepstral smoothing to find envelope
    """
    dft = numpy.fft.fft(x)

def toeplitz(arr):
    """
    creates a toeplitz matrix 2N-1 x 2N-1 where N is the size of arr
    """
    toep = numpy.zeros((len(arr)*2.-1,len(arr)))
    for i in numpy.arange(len(arr)):
        toep[:,i][i:(i+len(arr))] = arr
    return toep

def FLP(arr, n_coefs):
    R = numpy.zeros(n_coefs+1)
    for i in range(n_coefs+1):
        for j in range(len(arr)-i):
            R[i] += arr[j]*arr[j+i]
    A = numpy.zeros(n_coefs+1)
    A[0] = 1.
    
    Ek = R[0]
    #levinson Durbin recursion
    for i in range(n_coefs):
        lmd=0.0
        for j in range(i):
            lmd -= A[j]*R[i+1-j]
        lmd = lmd/Ek
        for k in range((i+1)/2):
            tmp = A[i+1-k]+lmd*A[k]
            A[k] = A[k]+lmd*A[i+1-k]
            A[i+1-k] = tmp
        Ek = Ek * (1.0-lmd*lmd)
        
    pred = arr.copy()
    for i in range(len(pred)):
        pred[i] = 0.0
        for j in range(n_coefs):
            pred[i] = pred[i]-A[j]*arr[i-1-j]
    return pred
            
# def top_peaks(x,n_peaks = 10):
    # ft = numpy.fft.fft(x)
    # peak_inds = numpy.argsort(-abs(ft))[:n_peaks*2]
    # ft_n = numpy.zeros(len(ft))
    # ft_n[peak_inds] = ft[peak_inds]
    # ift = numpy.fft.ifft(ft_n)
    # return ift