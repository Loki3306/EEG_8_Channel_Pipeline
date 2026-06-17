import numpy as np
from scipy.signal import gammatone, lfilter, butter, filtfilt, decimate

fs = 44100
audio = np.random.randn(fs * 10) * 10000
audio_float = audio.astype(np.float64)

# Try 'fir' mode
b_gt = gammatone(50, 'fir', fs=fs)
filtered_fir = lfilter(b_gt, [1.0], audio_float)
print("Max filtered FIR (50 Hz):", np.max(np.abs(filtered_fir)))
print("Has NaN FIR (50 Hz):", np.isnan(filtered_fir).any())

# Try downsampling first to 16000
audio_16k = decimate(audio_float, 3) # ~14700 Hz
b_gt, a_gt = gammatone(50, 'iir', fs=44100//3)
filtered_16k = lfilter(b_gt, a_gt, audio_16k)
print("Has NaN IIR (16k, 50 Hz):", np.isnan(filtered_16k).any())
