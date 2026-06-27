import os
import sys
import scipy.io as sio
import scipy.io.wavfile as wavfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import resample, butter, filtfilt, correlate

# Add repository root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from data.extract_gammatone_envelopes import extract_gammatone_envelopes
except ImportError as e:
    print(f"Could not import gammatone extraction: {e}")
    extract_gammatone_envelopes = None

def normalize_array(arr):
    arr = arr - arr.mean(axis=0, keepdims=True)
    scale = arr.std(axis=0, keepdims=True) + 1e-12
    return arr / scale

def main():
    print("\n" + "="*60)
    print("PHASE KUL-7: PAIR VERIFICATION AUDIT")
    print("="*60)
    
    if extract_gammatone_envelopes is None:
        print("ERROR: extract_gammatone_envelopes could not be imported!")
        return

    # Configuration
    mat_path = "/kaggle/input/datasets/lowk1ee/s1-klu/S1_KLU.mat"
    wav_dir = "/kaggle/input/datasets/lowk1ee/audio-klu"
    target_channels = ['Fp1', 'Fp2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8']
    fs_dtu = 64
    
    print("\nLoading KUL MAT file...")
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    trials = mat['trials'] if 'trials' in mat else mat['trial']
    
    # Pick 3 random trials
    np.random.seed(42)
    selected_trial_indices = np.random.choice(len(trials), 3, replace=False)
    
    def find_wav(name):
        for r, d, f in os.walk(wav_dir):
            if name in f: return os.path.join(r, name)
            if name+".wav" in f: return os.path.join(r, name+".wav")
        return None
        
    os.makedirs("analysis/figures/audit", exist_ok=True)
    
    for t_idx in selected_trial_indices:
        print(f"\n" + "-"*40)
        print(f"AUDITING TRIAL {t_idx+1}")
        print("-"*40)
        
        trial = trials[t_idx]
        eeg_data = trial.RawData.EegData
        fs_eeg = trial.FileHeader.SampleRate
        channel_names = [ch.Label for ch in trial.FileHeader.Channels]
        
        att_ear = trial.attended_ear
        stimuli = trial.stimuli
        
        print("1. METADATA TRACE")
        print(f"   Attended Ear from MAT: '{att_ear}'")
        print(f"   Stimulus Array (Left, Right): {stimuli}")
        
        att_wav_name = stimuli[0] if att_ear == 'L' else stimuli[1]
        unatt_wav_name = stimuli[1] if att_ear == 'L' else stimuli[0]
        
        print(f"   Pipeline Logic -> Attended Audio Assigned:   '{att_wav_name}'")
        print(f"   Pipeline Logic -> Unattended Audio Assigned: '{unatt_wav_name}'")
        
        if (att_ear == 'L' and att_wav_name != stimuli[0]) or (att_ear == 'R' and att_wav_name != stimuli[1]):
            print("   [!] ERROR: LOGICAL MISMATCH DETECTED IN AUDIO ASSIGNMENT!")
        else:
            print("   [+] Logic matches expected assignment.")
            
        att_wav_path = find_wav(str(att_wav_name))
        unatt_wav_path = find_wav(str(unatt_wav_name))
        
        if not att_wav_path or not unatt_wav_path:
            print("   [!] ERROR: WAV files not found on disk. Skipping signal audit.")
            continue
            
        # 2. Extract Signals
        print("\n2. SIGNAL EXTRACTION & PREPROCESSING")
        try:
            selected_indices = [channel_names.index(tc) if tc in channel_names else [c.upper() for c in channel_names].index(tc.upper()) for tc in target_channels]
        except ValueError as e:
            print(f"   [!] Missing channel: {e}")
            continue
            
        eeg_8 = eeg_data[:, selected_indices]
        
        # 1-8 Hz filter
        nyq = 0.5 * fs_eeg
        b, a = butter(2, [1.0/nyq, 8.0/nyq], btype='band')
        eeg_8_filt = filtfilt(b, a, eeg_8, axis=0)
        eeg_64 = resample(eeg_8_filt, int(len(eeg_8_filt) * fs_dtu / fs_eeg), axis=0)
        
        env_att = extract_gammatone_envelopes(att_wav_path, num_bands=28, low_freq=50, high_freq=8000, target_fs=fs_dtu)
        env_unatt = extract_gammatone_envelopes(unatt_wav_path, num_bands=28, low_freq=50, high_freq=8000, target_fs=fs_dtu)
        
        eeg_norm = normalize_array(eeg_64)
        env_att_norm = normalize_array(env_att.T).T
        env_unatt_norm = normalize_array(env_unatt.T).T
        
        # Ensure lengths match for correlation
        min_len = min(len(eeg_norm), env_att_norm.shape[1], env_unatt_norm.shape[1])
        eeg_norm = eeg_norm[:min_len]
        env_att_norm = env_att_norm[:, :min_len]
        env_unatt_norm = env_unatt_norm[:, :min_len]
        
        # Broadband proxy (mean across bands)
        att_broadband = env_att_norm.mean(axis=0)
        unatt_broadband = env_unatt_norm.mean(axis=0)
        
        # 3. Cross-Correlation
        print("\n3. CROSS-CORRELATION ANALYSIS (EEG vs Audio)")
        
        # We will check Fp1 (index 0) or average of all 8
        eeg_proxy = eeg_norm.mean(axis=1) # Mean across all 8 target channels for robustness
        
        # Compute valid cross-correlation
        max_lag_samples = int(1.5 * fs_dtu) # ±1.5 seconds
        lags = np.arange(-max_lag_samples, max_lag_samples + 1)
        
        xcorr_att = np.correlate(eeg_proxy, att_broadband, mode='full')
        xcorr_unatt = np.correlate(eeg_proxy, unatt_broadband, mode='full')
        
        # Center the cross-correlation
        center_idx = len(xcorr_att) // 2
        xcorr_att = xcorr_att[center_idx - max_lag_samples : center_idx + max_lag_samples + 1]
        xcorr_unatt = xcorr_unatt[center_idx - max_lag_samples : center_idx + max_lag_samples + 1]
        
        lags_ms = lags * (1000.0 / fs_dtu)
        
        # Plot Cross-Correlation
        plt.figure(figsize=(12, 6))
        plt.plot(lags_ms, xcorr_att, color='green', label=f'Attended ({att_wav_name})', alpha=0.8)
        plt.plot(lags_ms, xcorr_unatt, color='red', label=f'Unattended ({unatt_wav_name})', alpha=0.8)
        plt.axvline(0, color='gray', linestyle='dashed')
        plt.title(f"Trial {t_idx+1}: Cross-Correlation (EEG vs Audio Envelopes)\nExpect peak near 100-250ms for Attended")
        plt.xlabel("Lag (ms) [Positive = EEG lags Audio]")
        plt.ylabel("Cross-Correlation")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"analysis/figures/audit/trial_{t_idx+1}_cross_correlation.png")
        plt.close()
        
        print(f"   Saved Cross-Correlation plot to analysis/figures/audit/trial_{t_idx+1}_cross_correlation.png")
        
        # 4. Window Overlay
        print("\n4. WINDOW ALIGNMENT OVERLAY")
        # Plot 3 seconds of data at t = 10s
        start_sample = 10 * fs_dtu
        end_sample = start_sample + (3 * fs_dtu)
        
        t_axis = np.linspace(10, 13, 3 * fs_dtu)
        
        plt.figure(figsize=(14, 6))
        
        plt.subplot(2, 1, 1)
        plt.plot(t_axis, eeg_norm[start_sample:end_sample, 0], color='blue', label='EEG (Fp1)')
        plt.title(f"Trial {t_idx+1}: Raw Data Overlay (t=10s to 13s)")
        plt.ylabel("Amplitude (Z)")
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.plot(t_axis, att_broadband[start_sample:end_sample], color='green', label='Attended Env', alpha=0.8)
        plt.plot(t_axis, unatt_broadband[start_sample:end_sample], color='red', label='Unattended Env', alpha=0.8)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (Z)")
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"analysis/figures/audit/trial_{t_idx+1}_window_overlay.png")
        plt.close()
        print(f"   Saved Window Overlay plot to analysis/figures/audit/trial_{t_idx+1}_window_overlay.png")

    print("\nAudit Complete!")

if __name__ == "__main__":
    main()
