import sys
from pathlib import Path
import numpy as np
from scipy.signal import butter, filtfilt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from baselines.ridge_aad import (
    subject_files,
    load_subject_examples,
)

# Configuration matching the DTU paper reproduction
CHANNELS = list(range(64))
FS = 64
LAG_MS = 250
LAG_STEP_MS = 16
DECISION_WINDOW_SEC = 10
FIXED_LAMBDA = 1000.0
BP_LOWCUT = 1.0
BP_HIGHCUT = 8.0

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data, axis=0)
    return y

def normalize_array(arr):
    arr = arr - arr.mean(axis=0, keepdims=True)
    scale = arr.std(axis=0, keepdims=True) + 1e-12
    return arr / scale

def create_lagged_matrix(eeg, lags):
    samples = eeg.shape[1]
    channels = eeg.shape[0]
    blocks = []
    for lag in lags:
        if lag == 0:
            blocks.append(eeg.T)
        else:
            shifted = np.vstack([eeg.T[lag:, :], np.zeros((lag, channels))])
            blocks.append(shifted)
    return np.concatenate(blocks, axis=1)

def get_lag_offsets(lag_ms, lag_step_ms, fs):
    max_lag_samples = int(round((float(lag_ms) / 1000.0) * fs))
    step_samples = max(int(round((float(lag_step_ms) / 1000.0) * fs)), 1)
    offsets = list(range(0, max_lag_samples + 1, step_samples))
    if offsets[-1] != max_lag_samples:
        offsets.append(max_lag_samples)
    return sorted(set(offsets))

def evaluate_windows(pred, env_a, env_b, window_samples):
    num_correct = 0.0
    num_total = 0
    start = 0
    while start + window_samples <= len(pred):
        end = start + window_samples
        p = pred[start:end]
        ea = env_a[start:end]
        eb = env_b[start:end]
        
        std_p = np.std(p)
        if std_p < 1e-12:
            ca = 0.0
            cb = 0.0
        else:
            ca = np.corrcoef(p, ea)[0, 1]
            cb = np.corrcoef(p, eb)[0, 1]
            
        # env_a is ALWAYS the attended stream.
        if ca > cb:
            num_correct += 1.0
        elif ca == cb:
            num_correct += 0.5
                
        num_total += 1
        start += window_samples
    return num_correct, num_total

def main():
    print("===============================================================")
    print(" DTU SUBJECT 1 REPRODUCTION (FIXED TARGET ALIGNMENT)")
    print("===============================================================")
    
    paths = subject_files()
    if not paths:
        print("No subjects found. Exiting.")
        return
        
    # Pick S1
    s1_path = next(p for p in paths if p.stem.startswith("S1_"))
    print(f"Loading {s1_path.name}...")
    exs = load_subject_examples(s1_path)
    
    lags = get_lag_offsets(LAG_MS, LAG_STEP_MS, FS)
    print(f"Lags: 0 to {LAG_MS}ms ({len(lags)} steps at {FS}Hz)")
    print(f"Features: {len(CHANNELS)} channels x {len(lags)} lags = {len(CHANNELS) * len(lags)}")
    print(f"Decision Window: {DECISION_WINDOW_SEC}s")
    print(f"Fixed Lambda: {FIXED_LAMBDA}")
    
    window_samples = DECISION_WINDOW_SEC * FS
    
    xtx_list = []
    xty_list = []
    x_list = []
    env_a_list = []
    env_b_list = []
    n_samples_list = []
    
    print("Preprocessing trials...")
    for ex in exs:
        eeg = ex.eeg[:, CHANNELS].T
        
        # 1-8 Hz Bandpass
        eeg = butter_bandpass_filter(eeg, BP_LOWCUT, BP_HIGHCUT, FS)
        wav_a = butter_bandpass_filter(ex.wav_a.reshape(-1, 1), BP_LOWCUT, BP_HIGHCUT, FS).ravel()
        wav_b = butter_bandpass_filter(ex.wav_b.reshape(-1, 1), BP_LOWCUT, BP_HIGHCUT, FS).ravel()
        
        x = create_lagged_matrix(eeg, lags)
        x = normalize_array(x)
        
        env_a = normalize_array(wav_a.reshape(-1, 1)).ravel()
        env_b = normalize_array(wav_b.reshape(-1, 1)).ravel()
        
        # Target is ALWAYS env_a
        target_env = env_a
        
        mlen = min(x.shape[0], len(target_env))
        x = x[:mlen]
        env_a = env_a[:mlen]
        env_b = env_b[:mlen]
        target_env = target_env[:mlen]
        
        xtx_list.append(x.T @ x)
        xty_list.append(x.T @ target_env)
        
        x_list.append(x)
        env_a_list.append(env_a)
        env_b_list.append(env_b)
        n_samples_list.append(mlen)
        
    print(f"Processing {len(exs)} trials via LOTO...")
    
    subj_correct = 0.0
    subj_total = 0
    
    for test_idx in range(len(exs)):
        # Train
        train_indices = [i for i in range(len(exs)) if i != test_idx]
        
        outer_xtx = sum(xtx_list[i] for i in train_indices)
        outer_xty = sum(xty_list[i] for i in train_indices)
        outer_nsamp = sum(n_samples_list[i] for i in train_indices)
        
        reg = FIXED_LAMBDA * outer_nsamp * np.eye(outer_xtx.shape[0])
        final_weights = np.linalg.solve(outer_xtx + reg, outer_xty)
        
        # Evaluate Normal
        x_test = x_list[test_idx]
        pred = x_test @ final_weights
        nc, nt = evaluate_windows(pred, env_a_list[test_idx], env_b_list[test_idx], window_samples)
        
        subj_correct += nc
        subj_total += nt
        
    acc = subj_correct / subj_total
    
    print("\n===============================================================")
    print(" S1 FINAL RESULTS")
    print("===============================================================")
    print(f" Total Windows: {subj_total}")
    print(f" Correct Wins : {subj_correct}")
    print(f" Accuracy     : {acc*100:.2f}%")
    print("===============================================================")

if __name__ == "__main__":
    main()
