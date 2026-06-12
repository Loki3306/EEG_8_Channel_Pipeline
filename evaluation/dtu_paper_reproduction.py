import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from baselines.ridge_aad import (
    subject_files,
    load_subject_examples,
)

# Use all 64 channels
CHANNELS = list(range(64))
MAPPING = {1: "A", 2: "B"}
FS = 64
LAG_MS = 250
LAG_STEP_MS = 16
DECISION_WINDOW_SEC = 10  # 10s windowed evaluation
FIXED_LAMBDA = 1000.0  # Fixed lambda to drastically speed up execution


def normalize_array(arr):
    arr = arr - arr.mean(axis=0, keepdims=True)
    scale = arr.std(axis=0, keepdims=True) + 1e-12
    return arr / scale


def create_lagged_matrix(eeg, lags):
    """Create lagged design matrix from [Channels, Time] array."""
    samples = eeg.shape[1]
    channels = eeg.shape[0]
    
    blocks = []
    for lag in lags:
        if lag == 0:
            blocks.append(eeg.T)
        else:
            shifted = np.vstack([np.zeros((lag, channels)), eeg.T[:-lag, :]])
            blocks.append(shifted)
            
    return np.concatenate(blocks, axis=1)


def get_lag_offsets(lag_ms, lag_step_ms, fs):
    max_lag_samples = int(round((float(lag_ms) / 1000.0) * fs))
    step_samples = max(int(round((float(lag_step_ms) / 1000.0) * fs)), 1)
    offsets = list(range(0, max_lag_samples + 1, step_samples))
    if offsets[-1] != max_lag_samples:
        offsets.append(max_lag_samples)
    return sorted(set(offsets))


def evaluate_windows(pred, env_a, env_b, attended_stream, window_samples):
    """
    Splits prediction and targets into windows and computes accuracy.
    Returns: (num_correct, num_total)
    """
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
            
        if attended_stream == "A":
            if ca > cb:
                num_correct += 1.0
            elif ca == cb:
                num_correct += 0.5
        else:
            if cb > ca:
                num_correct += 1.0
            elif cb == ca:
                num_correct += 0.5
                
        num_total += 1
        start += window_samples
        
    return num_correct, num_total


def main():
    print("===============================================================")
    print(" DTU PAPER REPRODUCTION (64-Ch LOTO Ridge w/ Nested CV)")
    print("===============================================================")
    
    paths = subject_files()
    if not paths:
        print("No subjects found. Exiting.")
        return
        
    lags = get_lag_offsets(LAG_MS, LAG_STEP_MS, FS)
    print(f"Lags: 0 to {LAG_MS}ms ({len(lags)} steps at {FS}Hz)")
    print(f"Features: {len(CHANNELS)} channels x {len(lags)} lags = {len(CHANNELS) * len(lags)}")
    print(f"Decision Window: {DECISION_WINDOW_SEC}s")
    print(f"Lambda Grid: {LAMBDAS}")
    
    window_samples = DECISION_WINDOW_SEC * FS
    
    results_normal = []
    results_zero = []
    results_shuffle = []
    
    for path in paths:
        subject_id = path.stem.split('_')[0]
        exs = load_subject_examples(path)
        num_trials = len(exs)
        print(f"\n[Subject {subject_id}] Processing {num_trials} trials via nested LOTO...")
        
        # Precompute X^T X and X^T Y for all trials to massively speed up inner CV
        xtx_list = []
        xty_list = []
        x_list = []
        env_a_list = []
        env_b_list = []
        n_samples_list = []
        
        for ex in exs:
            eeg = ex.eeg[CHANNELS, :]
            x = create_lagged_matrix(eeg, lags)
            x = normalize_array(x)
            
            env_a = normalize_array(ex.wav_a.reshape(-1, 1)).ravel()
            env_b = normalize_array(ex.wav_b.reshape(-1, 1)).ravel()
            
            target_env = env_a if MAPPING[ex.label] == "A" else env_b
            
            mlen = min(x.shape[0], len(target_env))
            x = x[:mlen]
            target_env = target_env[:mlen]
            
            xtx_list.append(x.T @ x)
            xty_list.append(x.T @ target_env)
            
            x_list.append(x)
            env_a_list.append(env_a[:mlen])
            env_b_list.append(env_b[:mlen])
            n_samples_list.append(mlen)
            
        subj_normal_corr = 0.0
        subj_zero_corr = 0.0
        subj_shuffle_corr = 0.0
        subj_total_wins = 0
        
        # Shuffle indices for the "Shuffle EEG" test
        np.random.seed(42)
        shuffle_indices = np.random.permutation(num_trials)
        while np.any(shuffle_indices == np.arange(num_trials)):
            shuffle_indices = np.random.permutation(num_trials)
            
        # Outer LOTO loop
        for test_idx in range(num_trials):
            train_indices = [i for i in range(num_trials) if i != test_idx]
            
            # Train final model for this fold using fixed lambda
            outer_xtx = sum(xtx_list[i] for i in train_indices)
            outer_xty = sum(xty_list[i] for i in train_indices)
            outer_nsamp = sum(n_samples_list[i] for i in train_indices)
            
            reg = FIXED_LAMBDA * outer_nsamp * np.eye(outer_xtx.shape[0])
            final_weights = np.linalg.solve(outer_xtx + reg, outer_xty)
            
            # Evaluate Normal
            x_test_norm = x_list[test_idx]
            pred_norm = x_test_norm @ final_weights
            nc, nt = evaluate_windows(pred_norm, env_a_list[test_idx], env_b_list[test_idx], MAPPING[exs[test_idx].label], window_samples)
            subj_normal_corr += nc
            subj_total_wins += nt
            
            # Evaluate Zero
            pred_zero = np.zeros(x_test_norm.shape[0])
            nc_z, _ = evaluate_windows(pred_zero, env_a_list[test_idx], env_b_list[test_idx], MAPPING[exs[test_idx].label], window_samples)
            subj_zero_corr += nc_z
            
            # Evaluate Shuffle
            shuf_idx = shuffle_indices[test_idx]
            # Use the EEG (and x matrix) from the shuffled trial, but trim/pad to match length of target audio
            x_test_shuf = x_list[shuf_idx]
            mlen = min(x_test_shuf.shape[0], len(env_a_list[test_idx]))
            
            pred_shuf = x_test_shuf[:mlen] @ final_weights
            ea_shuf = env_a_list[test_idx][:mlen]
            eb_shuf = env_b_list[test_idx][:mlen]
            nc_s, _ = evaluate_windows(pred_shuf, ea_shuf, eb_shuf, MAPPING[exs[test_idx].label], window_samples)
            subj_shuffle_corr += nc_s
            
        acc_norm = subj_normal_corr / subj_total_wins
        acc_zero = subj_zero_corr / subj_total_wins
        acc_shuf = subj_shuffle_corr / subj_total_wins
        
        print(f"  -> Accuracy Normal : {acc_norm*100:.2f}%")
        print(f"  -> Accuracy Zero   : {acc_zero*100:.2f}%")
        print(f"  -> Accuracy Shuffle: {acc_shuf*100:.2f}%")
        
        results_normal.append(acc_norm)
        results_zero.append(acc_zero)
        results_shuffle.append(acc_shuf)
        
    print("\n===============================================================")
    print(" FINAL RESULTS (AVERAGE ACROSS 18 SUBJECTS)")
    print("===============================================================")
    print(f" Normal EEG  : {np.mean(results_normal)*100:.2f}%")
    print(f" Zero EEG    : {np.mean(results_zero)*100:.2f}%")
    print(f" Shuffle EEG : {np.mean(results_shuffle)*100:.2f}%")
    print("===============================================================")

if __name__ == "__main__":
    main()
