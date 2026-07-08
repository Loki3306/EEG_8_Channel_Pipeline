import os
import sys
import torch
import numpy as np
import scipy.signal
import scipy.io
import scipy.io.wavfile as wav
from pathlib import Path
from joblib import Parallel, delayed
import scipy.fft

def extract_multiband_envelopes(audio_data, sr=16000, target_sr=128, num_bands=16):
    """Extracts 16 individual speech envelopes using a Logarithmic Filterbank + Hilbert + Power Compression."""
    # 1. Safe Audio Normalization
    if np.issubdtype(audio_data.dtype, np.integer):
        audio = audio_data.astype(np.float32)
        audio /= np.iinfo(audio_data.dtype).max
    else:
        audio = audio_data.astype(np.float32)
        
    # 2. 16-Band Logarithmic Filterbank (150 Hz to 4000 Hz)
    edges = np.logspace(np.log10(150), np.log10(4000), num_bands + 1)
    
    N = len(audio)
    fast_N = scipy.fft.next_fast_len(N)
    
    multiband_env = np.zeros((num_bands, N), dtype=np.float32)
    
    for i in range(num_bands):
        low = edges[i]
        high = edges[i+1]
        
        # Bandpass filter for the current frequency band
        b, a = scipy.signal.butter(2, [low, high], btype='bandpass', fs=sr)
        band_audio = scipy.signal.filtfilt(b, a, audio)
        
        # Extract envelope for this specific band
        analytic = scipy.signal.hilbert(band_audio, N=fast_N)[:N]
        band_env = np.abs(analytic)
        
        # Power-law compression (biologically inspired)
        band_env = np.power(band_env, 0.6)
        
        multiband_env[i] = band_env
        
    # 3. Downsample to 128Hz
    env_ds = scipy.signal.resample_poly(multiband_env, target_sr, sr, axis=1) # [16, Time_ds]
    
    # 4. Low-pass filter at 8 Hz on the 128Hz signal (to match EEG cortical tracking)
    b_env, a_env = scipy.signal.butter(3, 8.0 / (target_sr / 2.0), btype='low')
    env_filt = scipy.signal.filtfilt(b_env, a_env, env_ds, axis=1)
    
    # 5. Normalize EACH band independently (Z-score)
    env_norm = (env_filt - np.mean(env_filt, axis=1, keepdims=True)) / (np.std(env_filt, axis=1, keepdims=True) + 1e-8)
    return env_norm # [16, Time_ds]

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
            
        # Extract pristine 16-band envelopes
        sr, wav_data = wav.read(wav_path)
        env_l = extract_multiband_envelopes(wav_data[:, 0], sr=sr, target_sr=128, num_bands=16) # [16, Time]
        env_r = extract_multiband_envelopes(wav_data[:, 1], sr=sr, target_sr=128, num_bands=16)
        
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
        
        min_len = min(trial_eeg_norm.shape[0], env_l.shape[1], env_r.shape[1])
        
        trials.append({
            'eeg': torch.from_numpy(trial_eeg_norm[:min_len].T).float(), # Transpose to (60, Time)
            'env_l': torch.from_numpy(env_l[:, :min_len]).float(), # [16, Time]
            'env_r': torch.from_numpy(env_r[:, :min_len]).float(),
            'meta': {'switch_points': switch_points}
        })
        
    return trials

if __name__ == "__main__":
    # --- MAIN EXECUTION ---
    data_root = '/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG'
    wav_dir = '/kaggle/input/datasets/lokeshgile/aasd-audio/Stimuli Audio'
    
    import glob
    # Process all subjects
    mat_files = glob.glob(os.path.join(data_root, 'S*', 'S*.mat'))
    mat_files.sort() # Ensure deterministic ordering
    
    cache_dir = Path('/kaggle/working/multiband_cache')
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating PRISTINE 16-Band Multiband Cache directly from WAV files in {cache_dir}...")
    
    def process_subject(mat_file):
        subj_id = os.path.basename(mat_file).split('.')[0]
        cache_path = cache_dir / f"{subj_id}_multiband.pt"
        
        if cache_path.exists():
            print(f"  [SKIPPED] {subj_id} is already cached!")
            return
            
        print(f"  [STARTING] {subj_id}...")
        try:
            trials = load_trials_from_raw(mat_file, wav_dir)
            torch.save({'raw': trials}, cache_path)
            print(f"  [SUCCESS] {subj_id} saved ({len(trials)} trials)")
        except Exception as e:
            print(f"  [ERROR] processing {subj_id}: {e}")
            
    # Process using all available CPU cores! (Kaggle has 4 cores)
    Parallel(n_jobs=-1)(delayed(process_subject)(mat_file) for mat_file in mat_files)
