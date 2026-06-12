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
    xtx = np.zeros((len(mean_x), len(mean_x)))
    xty = np.zeros(len(mean_x))
    
    for ex in train_exs:
        eeg = ex.eeg[CHANNELS, :]
        x = create_lagged_matrix(eeg, lags)
        x = (x - mean_x) / std_x
        
        env_a = normalize_envelope(ex.wav_a)
        env_b = normalize_envelope(ex.wav_b)
        
        target_env = env_a if MAPPING[ex.label] == "A" else env_b
        
        mlen = min(x.shape[0], len(target_env))
        x = x[:mlen]
        y = target_env[:mlen]
        
        xtx += x.T @ x
        xty += x.T @ y
        
    # Solve Ridge
    reg_matrix = RIDGE_LAMBDA * total_samples * np.eye(xtx.shape[0])
    weights = np.linalg.solve(xtx + reg_matrix, xty)
    
    return weights, mean_x, std_x


def evaluate_trial(ex, shuffled_ex, weights, mean_x, std_x, lags, mode="normal"):
    """
    Evaluates accuracy for a single trial.
    modes: 'normal', 'zero', 'shuffle'
    Returns score: 1.0 (correct), 0.5 (tie), 0.0 (incorrect)
    """
    if mode == "shuffle":
        # Use EEG from a different trial within the same subject
        eeg = shuffled_ex.eeg[CHANNELS, :]
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
    
    std_pred = np.std(pred)
    if std_pred < 1e-12:
        corr_a = 0.0
        corr_b = 0.0
    else:
        corr_a = np.corrcoef(pred, env_a)[0, 1]
        corr_b = np.corrcoef(pred, env_b)[0, 1]
        
    attended = MAPPING[ex.label]
    
    if attended == "A":
        if corr_a > corr_b:
            return 1.0
        elif corr_a == corr_b:
            return 0.5
    elif attended == "B":
        if corr_b > corr_a:
            return 1.0
        elif corr_b == corr_a:
            return 0.5
            
    return 0.0


def main():
    print("===============================================================")
    print(" DTU PAPER REPRODUCTION (64-Ch LOTO Ridge Baseline)")
    print("===============================================================")
    
    paths = subject_files()
    if not paths:
        print("No subjects found. Exiting.")
        return
        
    print(f"Loading {len(paths)} subjects...")
    lags = get_lag_offsets(LAG_MS, LAG_STEP_MS, FS)
    print(f"Using lag offsets (samples): {lags}")
    print(f"Total features per channel: {len(lags)}, Total parameters: {len(CHANNELS) * len(lags)}")
    
    results_normal = []
    results_zero = []
    results_shuffle = []
    
    for path in paths:
        subject_id = path.stem.split('_')[0]
        exs = load_subject_examples(path)
        num_trials = len(exs)
        print(f"\n[Subject {subject_id}] Processing {num_trials} trials via LOTO...")
        
        subj_normal = []
        subj_zero = []
        subj_shuffle = []
        
        # Pre-generate shuffle mapping (ensuring no trial maps to itself)
        np.random.seed(42)
        shuffle_indices = np.random.permutation(num_trials)
        while np.any(shuffle_indices == np.arange(num_trials)):
            shuffle_indices = np.random.permutation(num_trials)
            
        for test_idx in range(num_trials):
            # Leave one trial out
            train_exs = [ex for i, ex in enumerate(exs) if i != test_idx]
            test_ex = exs[test_idx]
            shuffled_ex = exs[shuffle_indices[test_idx]]
            
            # Train Ridge strictly on N-1 trials
            weights, mean_x, std_x = fit_ridge_model(train_exs, lags)
            
            # Evaluate test trial
            subj_normal.append(evaluate_trial(test_ex, shuffled_ex, weights, mean_x, std_x, lags, mode="normal"))
            subj_zero.append(evaluate_trial(test_ex, shuffled_ex, weights, mean_x, std_x, lags, mode="zero"))
            subj_shuffle.append(evaluate_trial(test_ex, shuffled_ex, weights, mean_x, std_x, lags, mode="shuffle"))
            
        subj_acc_normal = np.mean(subj_normal)
        subj_acc_zero = np.mean(subj_zero)
        subj_acc_shuffle = np.mean(subj_shuffle)
        
        print(f"  -> Accuracy Normal : {subj_acc_normal*100:.2f}%")
        print(f"  -> Accuracy Zero   : {subj_acc_zero*100:.2f}%")
        print(f"  -> Accuracy Shuffle: {subj_acc_shuffle*100:.2f}%")
        
        results_normal.append(subj_acc_normal)
        results_zero.append(subj_acc_zero)
        results_shuffle.append(subj_acc_shuffle)
        
    print("\n===============================================================")
    print(" FINAL RESULTS (AVERAGE ACROSS 18 SUBJECTS)")
    print("===============================================================")
    print(f" Normal EEG  : {np.mean(results_normal)*100:.2f}%")
    print(f" Zero EEG    : {np.mean(results_zero)*100:.2f}%")
    print(f" Shuffle EEG : {np.mean(results_shuffle)*100:.2f}%")
    print("===============================================================")

if __name__ == "__main__":
    main()
