import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from baselines.ridge_aad import (
    load_subject_examples,
    subject_files,
    fit_ridge,
    predict_envelope,
    target_envelope_raw,
    speech_envelope
)
from scipy.stats import pearsonr

SCREENING_SUBJECTS = ["S7_data_preproc", "S10_data_preproc", "S11_data_preproc", "S15_data_preproc"]
FS = 64
DECISION_WINDOW_SEC = 10

def evaluate_ridge_fold(weights, test_exs, mapping):
    window_samples = DECISION_WINDOW_SEC * FS
    n_correct = 0.0
    n_total = 0
    
    for ex in test_exs:
        # Predict envelope from EEG
        pred_env = predict_envelope(ex.eeg.T, weights, lags=16, fs=FS)
        
        # Get ground truth audio envelopes
        env_a = speech_envelope(ex.wav_a, fs=FS, normalize=True)
        env_b = speech_envelope(ex.wav_b, fs=FS, normalize=True)
        
        # Truncate to min length
        min_len = min(len(pred_env), len(env_a), len(env_b))
        pred_env = pred_env[:min_len]
        env_a = env_a[:min_len]
        env_b = env_b[:min_len]
        
        # Split into 10s chunks
        start = 0
        while start + window_samples <= min_len:
            end = start + window_samples
            chunk_pred = pred_env[start:end]
            chunk_a = env_a[start:end]
            chunk_b = env_b[start:end]
            
            # Use pearson correlation for Ridge (standard in AAD)
            r_a, _ = pearsonr(chunk_pred, chunk_a)
            r_b, _ = pearsonr(chunk_pred, chunk_b)
            
            # The model is trained to predict the attended envelope
            # so we check if correlation to attended > unattended
            target_is_a = (ex.label == 1) # Assumes mapping 1->A, 2->B
            # Wait, our mapping is defined. Let's just use the ex.label safely
            if target_is_a:
                correct = (r_a > r_b)
            else:
                correct = (r_b > r_a)
                
            if correct:
                n_correct += 1.0
            elif r_a == r_b:
                n_correct += 0.5
                
            n_total += 1
            start += window_samples
            
    return n_correct, n_total

def main():
    print(f"Running Ridge Baseline Within-Subject Screening on {SCREENING_SUBJECTS}")
    
    mapping = {1: "A", 2: "B"}
    all_paths = subject_files()
    paths = [p for p in all_paths if p.stem in SCREENING_SUBJECTS]
    
    overall_accs = []
    
    for subject_path in paths:
        print(f"\nEvaluating Within-Subject for: {subject_path.stem}")
        exs = load_subject_examples(subject_path)
        
        np.random.seed(42)
        np.random.shuffle(exs)
        
        k_folds = 5
        fold_size = len(exs) // k_folds
        
        sub_accs = []
        for k in range(k_folds):
            test_start = k * fold_size
            test_end = (k + 1) * fold_size if k < k_folds - 1 else len(exs)
            
            test_exs = exs[test_start:test_end]
            train_exs = exs[:test_start] + exs[test_end:]
            
            # Fit Ridge on training trials
            weights = fit_ridge(train_exs, mapping, lags=16, fs=FS, ridge_lambda=1.0)
            
            # Evaluate on test trials
            nc, nt = evaluate_ridge_fold(weights, test_exs, mapping)
            acc = nc / max(nt, 1)
            
            print(f"    Fold {k+1} Test Acc: {acc*100:.2f}%")
            sub_accs.append(acc)
            
        sub_mean = np.mean(sub_accs)
        print(f"  -> Ridge Subject Mean Acc : {sub_mean*100:.2f}%")
        overall_accs.append(sub_mean)
        
    print("\n" + "="*50)
    print("[RIDGE BASELINE WITHIN-SUBJECT SCREENING RESULTS]")
    print("="*50)
    print(f" Average Accuracy: {np.mean(overall_accs)*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()
