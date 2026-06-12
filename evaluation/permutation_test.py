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

# Configuration matching the reproduction script
CHANNELS = list(range(64))
MAPPING = {1: "A", 2: "B"}
FS = 64
LAG_MS = 250
LAG_STEP_MS = 16
DECISION_WINDOW_SEC = 10
FIXED_LAMBDA = 1000.0
BP_LOWCUT = 1.0
BP_HIGHCUT = 8.0
NUM_PERMUTATIONS = 1000


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
    print(" PERMUTATION SIGNIFICANCE TEST (1-8Hz Bandpass LOTO Ridge)")
    print(f" Running {NUM_PERMUTATIONS} Shuffles")
    print("===============================================================")
    
    paths = subject_files()
    if not paths:
        print("No subjects found. Exiting.")
        return
        
    lags = get_lag_offsets(LAG_MS, LAG_STEP_MS, FS)
    window_samples = DECISION_WINDOW_SEC * FS
    
    # We will accumulate global wins and total windows for TRUE and ALL PERMUTATIONS
    # This allows us to compute the global accuracy exactly as in the main script.
    global_true_wins = 0.0
    global_total_windows = 0
    
    # Store wins for each permutation: shape (NUM_PERMUTATIONS,)
    global_perm_wins = np.zeros(NUM_PERMUTATIONS)
    
    # Pre-generate 1000 random label assignments.
    # Each subject has ~60 trials. We'll generate a massive block of random labels per subject.
    # We use random coin flips (0 or 1) where 0=A, 1=B.
    # To be statistically strict, permutation tests often randomly flip labels exactly.
    np.random.seed(42)
    
    for path in paths:
        subject_id = path.stem.split('_')[0]
        exs = load_subject_examples(path)
        num_trials = len(exs)
        print(f"[{subject_id}] Precomputing {num_trials} trials...")
        
        xtx_list = []
        v_A_list = []
        v_B_list = []
        x_list = []
        env_a_list = []
        env_b_list = []
        n_samples_list = []
        true_labels = [] # 0 for A, 1 for B
        
        for ex in exs:
            eeg = ex.eeg[:, CHANNELS].T
            eeg = butter_bandpass_filter(eeg, BP_LOWCUT, BP_HIGHCUT, FS)
            wav_a = butter_bandpass_filter(ex.wav_a.reshape(-1, 1), BP_LOWCUT, BP_HIGHCUT, FS).ravel()
            wav_b = butter_bandpass_filter(ex.wav_b.reshape(-1, 1), BP_LOWCUT, BP_HIGHCUT, FS).ravel()
            
            x = create_lagged_matrix(eeg, lags)
            x = normalize_array(x)
            
            env_a = normalize_array(wav_a.reshape(-1, 1)).ravel()
            env_b = normalize_array(wav_b.reshape(-1, 1)).ravel()
            
            mlen = min(x.shape[0], len(env_a))
            x = x[:mlen]
            env_a = env_a[:mlen]
            env_b = env_b[:mlen]
            
            xtx_list.append(x.T @ x)
            v_A_list.append(x.T @ env_a)
            v_B_list.append(x.T @ env_b)
            
            x_list.append(x)
            env_a_list.append(env_a)
            env_b_list.append(env_b)
            n_samples_list.append(mlen)
            
            lbl = 0 if MAPPING[ex.label] == "A" else 1
            true_labels.append(lbl)
            
        true_labels = np.array(true_labels)
        
        # Precompute the inverse matrix for each LOTO fold.
        # This is the most expensive part, done exactly once per fold.
        inv_cache = []
        for test_idx in range(num_trials):
            train_indices = [i for i in range(num_trials) if i != test_idx]
            outer_xtx = sum(xtx_list[i] for i in train_indices)
            outer_nsamp = sum(n_samples_list[i] for i in train_indices)
            reg = FIXED_LAMBDA * outer_nsamp * np.eye(outer_xtx.shape[0])
            # Precompute inverse to make W calculation a simple dot product
            inv_matrix = np.linalg.inv(outer_xtx + reg)
            inv_cache.append(inv_matrix)
            
        print(f"[{subject_id}] Running {NUM_PERMUTATIONS} Permutations...")
        
        # Generate random labels for this subject for all permutations
        # shape: (NUM_PERMUTATIONS, num_trials)
        perm_labels = np.random.randint(0, 2, size=(NUM_PERMUTATIONS, num_trials))
        
        # To compute true accuracy, we treat it as "permutation index -1"
        all_labels = np.vstack([true_labels[np.newaxis, :], perm_labels])
        
        # Store wins for this subject
        subj_wins = np.zeros(NUM_PERMUTATIONS + 1)
        subj_total = 0
        
        # We can optimize by calculating V_train for all permutations simultaneously.
        # V_train = \sum_j V_j where V_j is V_A if label==0 else V_B.
        # v_A_list shape: (num_trials, features)
        V_A_arr = np.array(v_A_list)
        V_B_arr = np.array(v_B_list)
        
        # For each trial, shape (NUM_PERMS+1, features)
        # We can use np.where(all_labels == 0, V_A_arr, V_B_arr)
        # But V_A_arr is (trials, features), all_labels is (perms, trials)
        V_all = np.where(all_labels[:, :, np.newaxis] == 0, V_A_arr[np.newaxis, :, :], V_B_arr[np.newaxis, :, :])
        # V_all shape: (NUM_PERMS+1, num_trials, features)
        
        for test_idx in range(num_trials):
            # Sum over all train indices (axis=1)
            train_mask = np.ones(num_trials, dtype=bool)
            train_mask[test_idx] = False
            
            V_train = V_all[:, train_mask, :].sum(axis=1) # shape: (NUM_PERMS+1, features)
            
            # Compute W for all perms: W = V_train @ inv_matrix.T
            # inv_matrix shape: (features, features)
            # W shape: (NUM_PERMS+1, features)
            W_all = V_train @ inv_cache[test_idx].T
            
            x_test = x_list[test_idx]
            ea = env_a_list[test_idx]
            eb = env_b_list[test_idx]
            
            # Predict for all perms: pred = W @ x_test.T
            # pred shape: (NUM_PERMS+1, time)
            pred_all = W_all @ x_test.T
            
            # Evaluate each permutation
            for p in range(NUM_PERMUTATIONS+1):
                att = "A" if all_labels[p, test_idx] == 0 else "B"
                nc, nt = evaluate_windows(pred_all[p], ea, eb, att, window_samples)
                subj_wins[p] += nc
                if p == 0:
                    subj_total += nt # total is same for all permutations
                    
        global_true_wins += subj_wins[0]
        global_total_windows += subj_total
        global_perm_wins += subj_wins[1:]
        
        acc_true = subj_wins[0] / subj_total
        print(f"[{subject_id}] True Acc: {acc_true*100:.2f}% | Mean Perm Acc: {np.mean(subj_wins[1:])/subj_total*100:.2f}%")

    global_true_acc = global_true_wins / global_total_windows
    global_perm_accs = global_perm_wins / global_total_windows
    
    mean_null = np.mean(global_perm_accs)
    std_null = np.std(global_perm_accs)
    
    # Calculate exactly how many permutations scored higher than or equal to true accuracy
    better_perms = np.sum(global_perm_accs >= global_true_acc)
    p_value = better_perms / NUM_PERMUTATIONS
    
    print("\n===============================================================")
    print(" FINAL PERMUTATION TEST RESULTS")
    print("===============================================================")
    print(f" True Accuracy         : {global_true_acc*100:.2f}%")
    print(f" Null Distribution Mean: {mean_null*100:.2f}%")
    print(f" Null Distribution Std : {std_null*100:.2f}%")
    print(f" Exact P-Value         : {p_value:.4f}")
    if p_value < 0.05:
        print(" => SIGNIFICANT: The model extracts real EEG tracking signals.")
    else:
        print(" => NOT SIGNIFICANT: The accuracy is indistinguishable from random noise.")
    print("===============================================================")

if __name__ == "__main__":
    main()
