import os
import sys
import numpy as np
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(REPO_ROOT))

def test_windowed_extraction():
    # 1. Generate a mock 10-second audio signal at 48kHz (e.g. 1kHz sine wave + noise)
    fs = 48000
    t = np.linspace(0, 10, 10 * fs, endpoint=False)
    audio = np.sin(2 * np.pi * 1000 * t) + np.random.normal(0, 0.1, len(t))
    
    # 2. Offline Extraction (entire 10s file)
    from data.extract_subband_envelopes import extract_subband_envelopes
    import tempfile
    from scipy.io import wavfile
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        wav_path = f.name
    
    audio_int16 = np.int16(audio / np.max(np.abs(audio)) * 32767)
    wavfile.write(wav_path, fs, audio_int16)
    
    offline_env = extract_subband_envelopes(wav_path, target_fs=64)
    os.remove(wav_path)
    
    # 3. Windowed Extraction
    # Suppose at t=3.0s, we have access to [1.0s : 3.0s] (2 seconds of audio)
    # Let's extract that 2 second window using the EXACT same offline function
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        win_wav_path = f.name
        
    start_sec = 1.0
    end_sec = 3.0
    start_sample = int(start_sec * fs)
    end_sample = int(end_sec * fs)
    
    audio_win = audio_int16[start_sample:end_sample]
    wavfile.write(win_wav_path, fs, audio_win)
    
    win_env = extract_subband_envelopes(win_wav_path, target_fs=64)
    os.remove(win_wav_path)
    
    # 4. Compare the windowed envelope to the corresponding slice of the offline envelope
    # 2 seconds at 64Hz = 128 samples.
    # The corresponding offline slice is from 1.0s to 3.0s
    start_64 = int(start_sec * 64)
    end_64 = int(end_sec * 64)
    offline_slice = offline_env[:, start_64:end_64]
    
    print(f"Offline slice shape: {offline_slice.shape}")
    print(f"Windowed env shape: {win_env.shape}")
    
    diff = np.abs(offline_slice - win_env)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"Max absolute difference: {max_diff:.6f}")
    print(f"Mean absolute difference: {mean_diff:.6f}")
    
    # We expect some difference at the edges because sosfiltfilt and hilbert look at boundaries.
    # Let's see how much they differ in the middle of the window.
    trim = 16 # 0.25 seconds
    diff_mid = np.abs(offline_slice[:, trim:-trim] - win_env[:, trim:-trim])
    print(f"Max diff (trimmed 0.25s edges): {np.max(diff_mid):.6f}")
    print(f"Mean diff (trimmed 0.25s edges): {np.mean(diff_mid):.6f}")

if __name__ == "__main__":
    test_windowed_extraction()
