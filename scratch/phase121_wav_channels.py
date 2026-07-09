import os
import scipy.io.wavfile as wav
from scipy.signal import welch
import numpy as np

wav_dir = '/kaggle/input/datasets/lokeshgile/aasd-audio/Stimuli Audio'
if not os.path.exists(wav_dir):
    wav_dir = '/kaggle/input/aasd-audio/Stimuli Audio'

def detect_male_channel(wav_path):
    sr, data = wav.read(wav_path)
    f_L, Pxx_L = welch(data[:, 0], sr, nperseg=sr)
    f_R, Pxx_R = welch(data[:, 1], sr, nperseg=sr)
    valid_idx = (f_L >= 50) & (f_L <= 300)
    peak_L = f_L[valid_idx][np.argmax(Pxx_L[valid_idx])]
    peak_R = f_R[valid_idx][np.argmax(Pxx_R[valid_idx])]
    return 'L' if peak_L < peak_R else 'R'

print("Checking if Male speaker is ALWAYS on the Left channel...")
left_count = 0
right_count = 0

for i in range(1, 61):
    wav_path = os.path.join(wav_dir, f"mixed_{i:03d}.wav")
    if os.path.exists(wav_path):
        male_ch = detect_male_channel(wav_path)
        if male_ch == 'L': left_count += 1
        else: right_count += 1

print(f"Male on Left: {left_count} trials")
print(f"Male on Right: {right_count} trials")
