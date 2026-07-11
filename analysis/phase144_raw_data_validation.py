import os
import time
import numpy as np
import scipy.io
import scipy.io.wavfile as wav
import scipy.signal
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge

# -------------------------------------------------------------------------
# CONSTANTS & CONFIGURATION
# -------------------------------------------------------------------------
SR = 128
EAR_CHANNEL_INDICES = [23, 31, 32, 40, 14, 22, 41, 49]
PRE_SWITCH_SAMPLES = int(0.5 * SR)
POST_SWITCH_SAMPLES = int(1.0 * SR)
MIN_SEGMENT_SAMPLES = int(3.0 * SR)

LAG_MAX_MS = 400
LAG_MAX_SAMPLES = int((LAG_MAX_MS / 1000.0) * SR)

def compute_lagged_features(X, max_lag):
    """ Creates a lagged design matrix for TRF. X is [T, C] """
    T, C = X.shape
    X_lagged = np.zeros((T - max_lag, C * (max_lag + 1)))
    for lag in range(max_lag + 1):
        X_lagged[:, lag*C:(lag+1)*C] = X[max_lag-lag : T-lag, :]
    return X_lagged

def extract_broadband_envelope(audio_data, sr=16000, target_sr=128):
    """Extracts a simple, single broadband envelope using Hilbert transform."""
    if np.issubdtype(audio_data.dtype, np.integer):
        audio = audio_data.astype(np.float32)
        audio /= np.iinfo(audio_data.dtype).max
    else:
        audio = audio_data.astype(np.float32)
        
    N = len(audio)
    fast_N = scipy.fft.next_fast_len(N)
    
    # Simple Broadband Hilbert
    analytic = scipy.signal.hilbert(audio, N=fast_N)[:N]
    env = np.abs(analytic)
    
    # Downsample
    env_ds = scipy.signal.resample_poly(env, target_sr, sr)
    
    # Low-pass filter at 8 Hz
    b, a = scipy.signal.butter(3, 8.0 / (target_sr / 2.0), btype='low')
    env_filt = scipy.signal.filtfilt(b, a, env_ds)
    
    # Normalize
    env_norm = (env_filt - np.mean(env_filt)) / (np.std(env_filt) + 1e-8)
    return env_norm

def get_masks(sp, length):
    mask_true = np.zeros(length, dtype=np.float32)
    mask_valid = np.ones(length, dtype=bool)
    
    if len(sp) == 0:
        mask_true[:] = 1.0
        return mask_true, mask_valid
        
    current_state = 1.0 if sp[0][0] == 'R' else 0.0 
    last_idx = 0
    for spk, idx in sp:
        end_idx = min(idx, length)
        mask_true[last_idx:end_idx] = current_state
        current_state = 1.0 if spk == 'L' else 0.0
        last_idx = end_idx
        
        b_start = max(0, idx - PRE_SWITCH_SAMPLES)
        b_end = min(length, idx + POST_SWITCH_SAMPLES)
        mask_valid[b_start:b_end] = False
        
        if last_idx >= length:
            break
            
    if last_idx < length:
        mask_true[last_idx:] = current_state
        
    return mask_true, mask_valid

def process_raw_subject(mat_path, wav_dir):
    print(f"\nProcessing raw file: {mat_path}")
    mat = scipy.io.loadmat(mat_path, simplify_cells=True)
    events = mat['EEG_new']['event']
    data_all = mat['EEG_new']['data'] # Shape: (62 trials, 7680 timepoints, 60 channels)
    
    # Bandpass filter the EEG (1-8 Hz)
    b_eeg, a_eeg = scipy.signal.butter(4, [1.0/(128/2), 8.0/(128/2)], btype='band')
    
    segments = []
    
    for trial_idx in range(data_all.shape[0]):
        # Find Trial ID mapping to Audio
        trial_start_event = None
        expected_latency = trial_idx * 7680 + 1
        for ev in events:
            if abs(float(ev[1]) - expected_latency) < 0.5:
                trial_start_event = str(ev[0])
                break
                
        if not trial_start_event or not trial_start_event.isdigit(): continue
            
        audio_id = int(trial_start_event)
        wav_path = os.path.join(wav_dir, f"mixed_{audio_id:03d}.wav")
        if not os.path.exists(wav_path): continue
            
        # Extract RAW Broadband Envelope
        sr, wav_data = wav.read(wav_path)
        env_l = extract_broadband_envelope(wav_data[:, 0], sr=sr, target_sr=SR)
        env_r = extract_broadband_envelope(wav_data[:, 1], sr=sr, target_sr=SR)
        
        # Process RAW EEG
        trial_eeg = data_all[trial_idx]
        trial_eeg = trial_eeg - np.mean(trial_eeg, axis=1, keepdims=True) # CAR
        trial_eeg_filt = scipy.signal.filtfilt(b_eeg, a_eeg, trial_eeg, axis=0)
        trial_eeg_norm = (trial_eeg_filt - np.mean(trial_eeg_filt, axis=0, keepdims=True)) / (np.std(trial_eeg_filt, axis=0, keepdims=True) + 1e-8)
        
        # Subset to 8 Ear Channels
        eeg_ear = trial_eeg_norm[:, EAR_CHANNEL_INDICES] # [T, 8]
        
        # Switch Points
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
        min_len = min(eeg_ear.shape[0], len(env_l), len(env_r))
        mask_t, mask_v = get_masks(switch_points, min_len)
        
        # Extract Continuous Segments
        padded = np.concatenate([[False], mask_v, [False]])
        diff = np.diff(padded.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        for start, end in zip(starts, ends):
            if (end - start) < MIN_SEGMENT_SAMPLES:
                continue
            label = mask_t[start]
            X_seg = eeg_ear[start:end, :] # [T, C]
            target_env = env_l[start:end] if label == 1.0 else env_r[start:end]
            
            segments.append({
                'X': X_seg,
                'target_env': target_env
            })
            
    # H1: Lagged TRF
    if len(segments) < 2: return 0.0, 0.0, 1.0
    
    split_idx = int(len(segments) * 0.8)
    train_segs = segments[:split_idx]
    test_segs = segments[split_idx:]
    
    # Build Train
    X_train_list, Y_train_list = [], []
    for seg in train_segs:
        X_train_list.append(compute_lagged_features(seg['X'], LAG_MAX_SAMPLES))
        Y_train_list.append(seg['target_env'][LAG_MAX_SAMPLES:])
    X_train = np.vstack(X_train_list)
    Y_train = np.concatenate(Y_train_list)
    
    model = Ridge(alpha=1e3)
    model.fit(X_train, Y_train)
    
    # Build Test
    X_test_list, Y_test_list = [], []
    for seg in test_segs:
        X_test_list.append(compute_lagged_features(seg['X'], LAG_MAX_SAMPLES))
        Y_test_list.append(seg['target_env'][LAG_MAX_SAMPLES:])
    X_test = np.vstack(X_test_list)
    Y_test = np.concatenate(Y_test_list)
    
    Y_pred = model.predict(X_test)
    true_corr, _ = pearsonr(Y_test, Y_pred)
    
    # Permutation
    null_corrs = []
    np.random.seed(42)
    for _ in range(500):
        shift = np.random.randint(SR, len(Y_test) - SR)
        Y_test_shuf = np.roll(Y_test, shift)
        r, _ = pearsonr(Y_test_shuf, Y_pred)
        null_corrs.append(r)
        
    p_val = np.sum(np.array(null_corrs) >= true_corr) / 500
    mean_null = np.mean(null_corrs)
    
    return true_corr, mean_null, p_val

def main():
    print("=======================================================")
    print(" PHASE 144: RAW DATA VALIDATION (H1 BYPASSING CACHE)")
    print("=======================================================\n")
    
    data_root = '/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG'
    wav_dir = '/kaggle/input/datasets/lokeshgile/aasd-audio/Stimuli Audio'
    
    import glob
    mat_files = glob.glob(os.path.join(data_root, 'S*', 'S*.mat'))
    mat_files.sort()
    
    if not mat_files:
        print("ERROR: Raw data not found. Please run this script in the Kaggle environment with the AASD raw datasets attached.")
        return
        
    print(f"Testing H1 directly on RAW .mat and .wav for {mat_files[0]}")
    t_corr, null_corr, p_h1 = process_raw_subject(mat_files[0], wav_dir)
    
    print("\n--- Phase 144 Results ---")
    print(f"True Raw TRF Corr:  {t_corr:.4f}")
    print(f"Null Raw TRF Corr:  {null_corr:.4f}")
    print(f"P-Value:            {p_h1:.3f}")
    
    if p_h1 < 0.05:
        print("\nVERDICT: [SIGNAL FOUND!] The Multiband Cache corrupted the signal! The raw data contains the temporal envelope.")
    else:
        print("\nVERDICT: [HARDWARE LIMITATION] Even on pristine raw data, the 8-channel EEG does not contain temporal envelope tracking.")

if __name__ == '__main__':
    main()
