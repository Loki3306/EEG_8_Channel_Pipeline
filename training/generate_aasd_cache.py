import os
import sys
import torch
import numpy as np
import scipy.signal
import scipy.io
import scipy.io.wavfile as wav
from pathlib import Path

def extract_broadband_envelope(audio_data, sr=16000, target_sr=128):
    """Extracts the speech envelope using Hilbert + Power Compression."""
    # 1. Safe Audio Normalization
    if np.issubdtype(audio_data.dtype, np.integer):
        audio = audio_data.astype(np.float32)
        audio /= np.iinfo(audio_data.dtype).max
    else:
        audio = audio_data.astype(np.float32)
        
    # 2. Hilbert Envelope
    analytic = scipy.signal.hilbert(audio)
    env = np.abs(analytic)
    
    # 3. Power-law compression (Standard in KUL/DTU AAD pipelines)
    env = np.power(env, 0.6)
    
    # 4. Downsample BEFORE filtering (avoids numerical instability)
    # resample_poly reduces the ratio using GCD (target_sr up, sr down)
    env_ds = scipy.signal.resample_poly(env, target_sr, sr, axis=0)
    
    # 5. Low-pass filter at 8 Hz on the 128Hz signal (stable!)
    b_env, a_env = scipy.signal.butter(3, 8.0 / (target_sr / 2.0), btype='low')
    env_filt = scipy.signal.filtfilt(b_env, a_env, env_ds, axis=0)
    
    # 6. Normalize Envelope (Z-score)
    env_norm = (env_filt - np.mean(env_filt)) / (np.std(env_filt) + 1e-8)
    return env_norm

def load_trials_from_raw(mat_path, wav_dir):
    mat = scipy.io.loadmat(mat_path, simplify_cells=True)
    events = mat['EEG_new']['event']
    data_all = mat['EEG_new']['data'] # Shape: (62 trials, 7680 timepoints, 60 channels)
    
    # Bandpass filter the EEG (1-8 Hz)
    b_eeg, a_eeg = scipy.signal.butter(4, [1.0/(128/2), 8.0/(128/2)], btype='band')
    
    trials = []
    
    for trial_idx in range(data_all.shape[0]):
        trial_eeg = data_all[trial_idx] # (7680, 60)
        
        # EEG: Common Average Reference (CAR)
        trial_eeg = trial_eeg - np.mean(trial_eeg, axis=1, keepdims=True)
        
        # EEG: Bandpass Filter
        trial_eeg_filt = scipy.signal.filtfilt(b_eeg, a_eeg, trial_eeg, axis=0)
        
        # EEG: Channel-wise Normalization (Z-score)
        trial_eeg_norm = (trial_eeg_filt - np.mean(trial_eeg_filt, axis=0, keepdims=True)) / (np.std(trial_eeg_filt, axis=0, keepdims=True) + 1e-8)
        
        # Find Trial ID
        trial_start_event = None
        expected_latency = trial_idx * 7680 + 1
        for ev in events:
            # Safe floating point comparison for event alignment
            if abs(float(ev[1]) - expected_latency) < 0.5:
                trial_start_event = str(ev[0])
                break
                
        if not trial_start_event or not trial_start_event.isdigit(): continue
            
        audio_id = int(trial_start_event)
        wav_path = os.path.join(wav_dir, f"mixed_{audio_id:03d}.wav")
        if not os.path.exists(wav_path): continue
            
        # Extract pristine envelopes
        sr, wav_data = wav.read(wav_path)
        env_l = extract_broadband_envelope(wav_data[:, 0], sr=sr, target_sr=128)
        env_r = extract_broadband_envelope(wav_data[:, 1], sr=sr, target_sr=128)
        
        # Find Switch Points
        epoch_start_lat = trial_idx * 7680 + 1
        switch_points = []
        for ev in events:
            t_str = str(ev[0])
            if t_str in ['179', '184']:
                abs_lat = float(ev[1])
                if epoch_start_lat <= abs_lat < epoch_start_lat + 7680:
                    rel_lat = abs_lat - epoch_start_lat
                    idx_128 = max(0, int(round(rel_lat)))
                    switch_points.append(('L' if t_str == '179' else 'R', idx_128))
                    
        switch_points.sort(key=lambda x: x[1])
        
        min_len = min(trial_eeg_norm.shape[0], env_l.shape[0], env_r.shape[0])
        
        trials.append({
            'eeg': torch.from_numpy(trial_eeg_norm[:min_len].T).float(), # Transpose to (60, Time)
            'env_l': torch.from_numpy(env_l[:min_len]).float(),
            'env_r': torch.from_numpy(env_r[:min_len]).float(),
            'meta': {'switch_points': switch_points}
        })
        
    return trials

if __name__ == "__main__":
    # --- MAIN EXECUTION ---
    data_root = '/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG'
    wav_dir = '/kaggle/input/datasets/lokeshgile/aasd-audio/Stimuli Audio'
    
    mat_files = []
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file.endswith('.mat') and not file.startswith('._'):
                mat_files.append(os.path.join(root, file))
    mat_files.sort()
    
    cache_dir = Path('/kaggle/working/eeg_cache')
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating PRISTINE cache directly from WAV files in {cache_dir}...")
    
    for mat_file in mat_files:
        subj_id = os.path.basename(mat_file).split('.')[0]
        cache_path = cache_dir / f"{subj_id}_processed.pt"
        
        if not cache_path.exists():
            print(f"  Processing {subj_id}...")
            try:
                trials = load_trials_from_raw(mat_file, wav_dir)
                torch.save({'raw': trials}, cache_path)
                print(f"    -> Saved {subj_id}_processed.pt (extracted {len(trials)} trials)")
            except Exception as e:
                print(f"    -> ERROR processing {subj_id}: {e}")
        else:
            print(f"  {subj_id} already cached.")
    
    print("\nDone! You can now run Phase 47.")
