import os
import argparse
import scipy.io as sio
import scipy.io.wavfile as wavfile
import numpy as np
from scipy.signal import resample, butter, filtfilt

def erb_space(low_freq, high_freq, num_bands):
    erb_low = 21.4 * np.log10(4.37 * low_freq / 1000 + 1)
    erb_high = 21.4 * np.log10(4.37 * high_freq / 1000 + 1)
    erb_points = np.linspace(erb_low, erb_high, num_bands)
    cf = (10 ** (erb_points / 21.4) - 1) * 1000 / 4.37
    return cf

def get_erb_bands(cfs):
    # ERB bandwidth = 24.7 * (4.37 * F / 1000 + 1)
    bws = 24.7 * (4.37 * cfs / 1000 + 1)
    lows = cfs - bws / 2
    highs = cfs + bws / 2
    return lows, highs

def apply_bandpass(data, low, high, fs, order=2):
    nyq = 0.5 * fs
    low = max(low / nyq, 0.001)
    high = min(high / nyq, 0.999)
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def extract_28_band_envelope(audio_data, fs_in, fs_out=64, num_bands=28):
    cfs = erb_space(50, 8000, num_bands)
    lows, highs = get_erb_bands(cfs)
    
    envelopes = []
    # 1. Bandpass into 28 bands
    for i in range(num_bands):
        band_audio = apply_bandpass(audio_data, lows[i], highs[i], fs_in)
        # 2. Extract Envelope (absolute value)
        env = np.abs(band_audio)
        # 3. Compression (DTU matlab uses .^.3)
        env = env ** 0.3
        envelopes.append(env)
        
    envelopes = np.array(envelopes) # Shape: (28, T_in)
    
    # 4. Downsample to 64 Hz
    num_samples_out = int(envelopes.shape[1] * fs_out / fs_in)
    envelopes_out = resample(envelopes, num_samples_out, axis=1) # Shape: (28, T_out)
    
    return envelopes_out

def main():
    print("\n" + "="*50)
    print("PHASE KUL-2.5: RECONSTRUCT DTU 28-BAND AUDIO REPRESENTATION")
    print("="*50)
    
    mat_path = "/kaggle/input/datasets/lowk1ee/s1-klu/S1_KLU.mat"
    wav_dir = "/kaggle/input/datasets/lowk1ee/audio-klu"
    
    print("1. Loading S1 Trial 0 EEG...")
    try:
        mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        if 'trials' in mat: trials = mat['trials']
        elif 'trial' in mat: trials = mat['trial']
        else:
            print("No trials found.")
            return
        trial = trials[0]
    except Exception as e:
        print(f"Error loading MAT: {e}")
        return
        
    eeg_data = trial.RawData.EegData
    fs_eeg = trial.FileHeader.SampleRate
    channel_names = [ch.Label for ch in trial.FileHeader.Channels]
    
    target_channels = ['Fp1', 'Fp2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8']
    selected_indices = [channel_names.index(tc) if tc in channel_names else [c.upper() for c in channel_names].index(tc.upper()) for tc in target_channels]
    
    eeg_8 = eeg_data[:, selected_indices]
    fs_dtu = 64
    eeg_64 = resample(eeg_8, int(len(eeg_8) * fs_dtu / fs_eeg), axis=0)
    
    att_ear = trial.attended_ear
    stimuli = trial.stimuli
    att_wav_name = stimuli[0] if att_ear == 'L' else stimuli[1]
    unatt_wav_name = stimuli[1] if att_ear == 'L' else stimuli[0]
    
    print("2. Locating Audio Files...")
    def find_wav(name):
        for root, _, files in os.walk(wav_dir):
            if name in files: return os.path.join(root, name)
            if name+".wav" in files: return os.path.join(root, name+".wav")
        return None
        
    att_wav_path = find_wav(str(att_wav_name))
    unatt_wav_path = find_wav(str(unatt_wav_name))
    
    if not att_wav_path or not unatt_wav_path:
        print("ERROR: Could not find WAV files.")
        return
        
    fs_att, audio_att = wavfile.read(att_wav_path)
    fs_unatt, audio_unatt = wavfile.read(unatt_wav_path)
    
    if len(audio_att.shape) > 1: audio_att = audio_att.mean(axis=1)
    if len(audio_unatt.shape) > 1: audio_unatt = audio_unatt.mean(axis=1)
    
    print("3. Generating 28-Band ERB Envelopes...")
    print(f"   Processing Attended Audio ({len(audio_att)} samples @ {fs_att}Hz)")
    env_att = extract_28_band_envelope(audio_att, fs_att, fs_out=64, num_bands=28)
    
    print(f"   Processing Unattended Audio ({len(audio_unatt)} samples @ {fs_unatt}Hz)")
    env_unatt = extract_28_band_envelope(audio_unatt, fs_unatt, fs_out=64, num_bands=28)
    
    print(f"\n   -> Attended Audio Shape  : {env_att.shape}")
    print(f"   -> Unattended Audio Shape: {env_unatt.shape}")
    
    print("\n4. Aligning and Windowing Tensors...")
    min_len = min(len(eeg_64), env_att.shape[1], env_unatt.shape[1])
    
    eeg_64 = eeg_64[:min_len]
    env_att = env_att[:, :min_len]
    env_unatt = env_unatt[:, :min_len]
    
    window_sec, stride_sec = 3.0, 1.5
    win_samples = int(window_sec * fs_dtu)
    stride_samples = int(stride_sec * fs_dtu)
    
    eeg_windows, att_windows, unatt_windows = [], [], []
    for start in range(0, min_len - win_samples + 1, stride_samples):
        eeg_windows.append(eeg_64[start:start+win_samples])
        att_windows.append(env_att[:, start:start+win_samples])
        unatt_windows.append(env_unatt[:, start:start+win_samples])
        if len(eeg_windows) >= 10: break
        
    X_eeg = np.array(eeg_windows)
    X_att = np.array(att_windows)
    X_unatt = np.array(unatt_windows)
    
    print("\n5. Verification Results")
    for i in range(3):
        print(f"   Window {i}:")
        print(f"     EEG        : {X_eeg[i].shape}")
        print(f"     Attended   : {X_att[i].shape}")
        print(f"     Unattended : {X_unatt[i].shape}")
        
    print("\nFINAL VERDICT:")
    if X_att[0].shape == (28, 192) and X_eeg[0].shape == (192, 8):
        print("SUCCESS! The KUL preprocessing pipeline now generates the exact 28-band tensor geometry required by ContrastiveMatchNet.")
    else:
        print("FAILED. Tensor geometries do not match expectations.")

if __name__ == "__main__":
    main()
