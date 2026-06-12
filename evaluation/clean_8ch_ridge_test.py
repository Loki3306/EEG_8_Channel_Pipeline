import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from baselines.ridge_aad import (
    subject_files,
    load_subject_examples,
    iter_leave_one_subject_out,
)

CHANNELS = [12, 14, 16, 22, 50, 52, 54, 60]
MAPPING = {1: "A", 2: "B"}
FS = 64
LAG_MS = 250
LAG_STEP_MS = 16
RIDGE_LAMBDA = 1.0


def normalize_envelope(wav):
    wav = wav.ravel().astype(float)
    wav = wav - wav.mean()
    scale = wav.std() + 1e-12
    return wav / scale


def create_lagged_matrix(eeg, lags):
    """Create lagged design matrix from [Channels, Time] array."""
    samples = eeg.shape[1]
    channels = eeg.shape[0]
    
    # Pre-allocate blocks
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


def fit_ridge_model(train_exs, lags):
    """Computes Ridge weights using the exact normal equation."""
    # First pass: find feature means and stds for standardization
    sum_x = None
    sum_sq_x = None
    total_samples = 0
    
    print("  -> Computing feature statistics...")
    for ex in train_exs:
        eeg = ex.eeg[CHANNELS, :]
        x = create_lagged_matrix(eeg, lags)
        if sum_x is None:
            sum_x = np.zeros(x.shape[1])
            sum_sq_x = np.zeros(x.shape[1])
            
        sum_x += x.sum(axis=0)
        sum_sq_x += np.square(x).sum(axis=0)
        total_samples += x.shape[0]
        
    mean_x = sum_x / total_samples
    var_x = (sum_sq_x / total_samples) - np.square(mean_x)
    std_x = np.sqrt(np.maximum(var_x, 1e-12))
    
    # Second pass: compute X^T X and X^T Y
    print("  -> Accumulating sufficient statistics...")
    xtx = np.zeros((len(mean_x), len(mean_x)))
    xty = np.zeros(len(mean_x))
    
    for ex in train_exs:
        eeg = ex.eeg[CHANNELS, :]
        x = create_lagged_matrix(eeg, lags)
        x = (x - mean_x) / std_x
        
        env_a = normalize_envelope(ex.wav_a)
        env_b = normalize_envelope(ex.wav_b)
        
        target_env = env_a if MAPPING[ex.label] == "A" else env_b
        
        # Match lengths (lagging might not alter length here since we zero pad, but safety first)
        mlen = min(x.shape[0], len(target_env))
        x = x[:mlen]
        y = target_env[:mlen]
        
        xtx += x.T @ x
        xty += x.T @ y
        
    # Solve Ridge
    print("  -> Solving normal equations...")
    reg_matrix = RIDGE_LAMBDA * total_samples * np.eye(xtx.shape[0])
    weights = np.linalg.solve(xtx + reg_matrix, xty)
    
    return weights, mean_x, std_x


def evaluate_test_set(test_exs, weights, mean_x, std_x, lags, mode="normal"):
    """
    Evaluates accuracy.
    modes: 'normal', 'zero', 'shuffle'
    """
    n_correct = 0
    
    # If shuffle, we need to randomly permute the EEGs
    if mode == "shuffle":
        all_eegs = [ex.eeg for ex in test_exs]
        np.random.seed(42)
        np.random.shuffle(all_eegs)
        
    for i, ex in enumerate(test_exs):
        if mode == "shuffle":
            eeg = all_eegs[i][CHANNELS, :]
        else:
            eeg = ex.eeg[CHANNELS, :]
            
        if mode == "zero":
            eeg = np.zeros_like(eeg)
            
        x = create_lagged_matrix(eeg, lags)
        x = (x - mean_x) / std_x
        
        pred = x @ weights
        
        env_a = normalize_envelope(ex.wav_a)
        env_b = normalize_envelope(ex.wav_b)
        
        mlen = min(len(pred), len(env_a), len(env_b))
        pred = pred[:mlen]
        env_a = env_a[:mlen]
        env_b = env_b[:mlen]
        
        # If prediction is perfectly constant, corrcoef will throw warning/NaN
        std_pred = np.std(pred)
        if std_pred < 1e-12:
            # Fallback for constant prediction
            corr_a = 0.0
            corr_b = 0.0
        else:
            corr_a = np.corrcoef(pred, env_a)[0, 1]
            corr_b = np.corrcoef(pred, env_b)[0, 1]
            
        attended = MAPPING[ex.label]
        if attended == "A" and corr_a > corr_b:
            n_correct += 1
        elif attended == "B" and corr_b > corr_a:
            n_correct += 1
            
    return n_correct / len(test_exs)


def main():
    print("===============================================================")
    print(" CLEAN 8-CHANNEL RIDGE BASELINE SANITY SUITE")
    print("===============================================================")
    
    paths = subject_files()
    if not paths:
        print("No subjects found. Exiting.")
        return
        
    print(f"Loading {len(paths)} subjects...")
    subject_examples = {str(p): load_subject_examples(p) for p in paths}
    
    lags = get_lag_offsets(LAG_MS, LAG_STEP_MS, FS)
    print(f"Using lag offsets (samples): {lags}")
    print(f"Total features per channel: {len(lags)}, Total parameters: {len(CHANNELS) * len(lags)}")
    
    results_normal = []
    results_zero = []
    results_shuffle = []
    
    for fold_idx, (held_out, train_paths) in enumerate(iter_leave_one_subject_out(paths), start=1):
        print(f"\n[Fold {fold_idx}/18] Held out: {held_out.stem}")
        
        train_exs = []
        for tp in train_paths:
            train_exs.extend(subject_examples[str(tp)])
        test_exs = subject_examples[str(held_out)]
        
        weights, mean_x, std_x = fit_ridge_model(train_exs, lags)
        
        acc_normal = evaluate_test_set(test_exs, weights, mean_x, std_x, lags, mode="normal")
        acc_zero = evaluate_test_set(test_exs, weights, mean_x, std_x, lags, mode="zero")
        acc_shuffle = evaluate_test_set(test_exs, weights, mean_x, std_x, lags, mode="shuffle")
        
        print(f"  Accuracy Normal : {acc_normal*100:.2f}%")
        print(f"  Accuracy Zero   : {acc_zero*100:.2f}%")
        print(f"  Accuracy Shuffle: {acc_shuffle*100:.2f}%")
        
        results_normal.append(acc_normal)
        results_zero.append(acc_zero)
        results_shuffle.append(acc_shuffle)
        
    print("\n===============================================================")
    print(" FINAL RESULTS (OVERALL MEAN ACCURACY)")
    print("===============================================================")
    print(f" Normal EEG  : {np.mean(results_normal)*100:.2f}%")
    print(f" Zero EEG    : {np.mean(results_zero)*100:.2f}%")
    print(f" Shuffle EEG : {np.mean(results_shuffle)*100:.2f}%")
    print("===============================================================")

if __name__ == "__main__":
    main()
