import numpy as np
from matplotlib import pyplot as plt
from time import time
from scipy.fft import fft, fftfreq
from math import log2
SAMPLE_RATE = 44100  # Hertz
DURATION = 5  # Seconds

'''参考文献
https://zhuanlan.zhihu.com/p/407885496
https://realpython.com/python-scipy-fft/'''

def generate_sine_wave(freq, sample_rate, duration):
    x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    frequencies = x * freq
    y = np.sin((2 * np.pi) * frequencies)
    return x, y

def my_dft(tone : np.ndarray) -> np.ndarray:
    #实现一维傅里叶变换
    N = len(tone)
    dft_result = np.zeros(N, dtype = complex)
    for k in range(N):
        for i in range(N):
            dft_result[k] += tone[i] * np.exp(-2j * np.pi * k * i / N)
    return dft_result        

def my_fft(tone: np.ndarray) -> np.ndarray:
    #实现一维快速傅里叶变换
    N = len(tone)
    if N == 0:
        return np.array([])
    if N & (N - 1) != 0:  # Check if N is a power of 2
        raise ValueError("Input length must be a power of 2")

    p = int(log2(N))
    A1 = tone.astype(np.complex128)
    A2 = np.zeros(N, dtype=np.complex128)
    omegas = np.exp(-2j * np.pi * np.arange(N // 2) / N)

    for q in range(1, p + 1):
        if q % 2 == 1:
            for k in range(2 ** (p - q)):
                for j in range(2 ** (q - 1)):
                    A2[k * 2 ** q + j] = A1[k * 2 ** (q - 1) + j] + A1[k * 2 ** (q - 1) + j + 2 ** (p - 1)]
                    A2[k * 2 ** q + j + 2 ** (q - 1)] = (A1[k * 2 ** (q - 1) + j] - A1[k * 2 ** (q - 1) + j + 2 ** (p - 1)]) * omegas[k * 2 ** (q - 1)]
        else:
            for k in range(2 ** (p - q)):
                for j in range(2 ** (q - 1)):
                    A1[k * 2 ** q + j] = A2[k * 2 ** (q - 1) + j] + A2[k * 2 ** (q - 1) + j + 2 ** (p - 1)]
                    A1[k * 2 ** q + j + 2 ** (q - 1)] = (A2[k * 2 ** (q - 1) + j] - A2[k * 2 ** (q - 1) + j + 2 ** (p - 1)]) * omegas[k * 2 ** (q - 1)]

    return A2 if p % 2 == 1 else A1

_, nice_tone = generate_sine_wave(400, SAMPLE_RATE, DURATION)
_, noise_tone = generate_sine_wave(4000, SAMPLE_RATE, DURATION)
noise_tone = noise_tone * 0.3
mixed_tone = nice_tone + noise_tone
normalized_tone = np.int16((mixed_tone / mixed_tone.max()) * 32767)
plt.plot(normalized_tone[:1000])
plt.title('waveform')
plt.show()

time0 = time()
yf0 = fft(normalized_tone[:1024])
time1 = time()
yf1 = my_dft(normalized_tone[:1024])
time2 = time()
yf2 = my_fft(normalized_tone[:1024])
time3 = time()
xf = fftfreq(1024, 1 / SAMPLE_RATE)

print(f'running time of sci.fft is {time1 - time0}')
print(f'running time of my_dft is {time2 - time1}')
print(f'running time of my_fft is {time3 - time2}')

fig, axs = plt.subplots(1, 3, figsize = (10, 3)) # 通过figsize调整图大小
plt.subplots_adjust(wspace = 0.2, hspace = 0.2) # 通过wspace和hspace调整子图间距
plt.subplot(131) 
plt.plot(xf, np.abs(yf0)) 
plt.grid() 
plt.title(f'sci.fft')
plt.subplot(132) 
plt.plot(xf, np.abs(yf1)) 
plt.grid() 
plt.title(f'my_dft')
plt.subplot(133)
plt.plot(xf, np.abs(yf2)) 
plt.grid()
plt.title(f'my_fft')
plt.show()