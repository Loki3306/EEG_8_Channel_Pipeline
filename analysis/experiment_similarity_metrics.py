import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from sklearn.cross_decomposition import CCA
from sklearn.metrics import mutual_info_score
import time
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.kul_cached_dataset import KULCachedLoader

def pearson_np(x, y, eps=1e-8):
    x_mean = x.mean()
    y_mean = y.mean()
    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
    return num / (den + eps)

def cosine_np(x, y, eps=1e-8):
    return np.sum(x * y) / (np.linalg.norm(x) * np.linalg.norm(y) + eps)

def lagged_pearson(x, y, max_lag=5):
    """Computes max pearson correlation over shifts -max_lag to +max_lag"""
    best_corr = -1.0
    # x and y are 1D arrays
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            x_shift = x[:lag]
            y_shift = y[-lag:]
        elif lag > 0:
            x_shift = x[lag:]
            y_shift = y[:-lag]
        else:
            x_shift = x
            y_shift = y
        c = pearson_np(x_shift, y_shift)
        if c > best_corr:
            best_corr = c
    return best_corr

def fast_mi(x, y, bins=20):
    c, _, _ = np.histogram2d(x, y, bins=bins)
    return mutual_info_score(None, None, contingency=c)

def evaluate_window(pred, target_a, target_b):
    """
    pred, target_a, target_b: (28, Time)
    Returns dictionary of similarities for Attended and Unattended
    """
    num_bands = pred.shape[0]
    
    # 1. Pearson (Average over bands)
    p_a = np.mean([pearson_np(pred[i], target_a[i]) for i in range(num_bands)])
    p_b = np.mean([pearson_np(pred[i], target_b[i]) for i in range(num_bands)])
    
    # 2. Cosine (Average over bands)
    c_a = np.mean([cosine_np(pred[i], target_a[i]) for i in range(num_bands)])
    c_b = np.mean([cosine_np(pred[i], target_b[i]) for i in range(num_bands)])
    
    # 3. Lagged Cross-Correlation (Max average over lags)
    lc_a = np.mean([lagged_pearson(pred[i], target_a[i], max_lag=5) for i in range(num_bands)])
    lc_b = np.mean([lagged_pearson(pred[i], target_b[i], max_lag=5) for i in range(num_bands)])
    
    # 4. CCA (1 component on all 28 bands)
    try:
        cca_a = CCA(n_components=1)
        pred_a_c, target_a_c = cca_a.fit_transform(pred.T, target_a.T)
        cca_score_a = pearson_np(pred_a_c[:, 0], target_a_c[:, 0])
        
        cca_b = CCA(n_components=1)
        pred_b_c, target_b_c = cca_b.fit_transform(pred.T, target_b.T)
        cca_score_b = pearson_np(pred_b_c[:, 0], target_b_c[:, 0])
    except Exception:
        # Fallback if CCA fails due to collinearity
        cca_score_a = 0.0
        cca_score_b = 0.0
        
    # 5. Fast Mutual Information (On mean envelope)
    pred_1d = pred.mean(axis=0)
    t_a_1d = target_a.mean(axis=0)
    t_b_1d = target_b.mean(axis=0)
    
    mi_a = fast_mi(pred_1d, t_a_1d, bins=20)
    mi_b = fast_mi(pred_1d, t_b_1d, bins=20)
    
    return {
        "Pearson": (p_a, p_b),
        "Cosine": (c_a, c_b),
        "Lagged_Corr": (lc_a, lc_b),
        "CCA": (cca_score_a, cca_score_b),
        "Mutual_Info": (mi_a, mi_b)
    }

def main():
    print("================================================================")
    print("             SIMILARITY METRIC EVALUATION (RIDGE)               ")
    print("================================================================")
    
    loader = KULCachedLoader(REPO_ROOT / "data" / "processed_kul")
    try:
        all_subject_data = loader.load_all()
    except FileNotFoundError:
        print("KUL Cache not found. Please run preprocessing first.")
        return
        
    num_channels = 8
    num_lags = 17 
    feature_count = num_channels * num_lags
    num_bands = 28
    
    subject_xtx = {}
    subject_xty = {}
    
    global_xtx = np.zeros((feature_count, feature_count), dtype=float)
    global_xty = np.zeros((feature_count, num_bands), dtype=float)
    
    print("\nPhase 1: Accumulating Global Ridge Matrices...")
    for sub, trials in all_subject_data.items():
        s_xtx = np.zeros((feature_count, feature_count), dtype=float)
        s_xty = np.zeros((feature_count, num_bands), dtype=float)
        
        for t in trials:
            eeg_np = t["eeg"].numpy()
            a_np = t["audio_a"].numpy()
            
            e_mean = eeg_np.mean(axis=1, keepdims=True)
            e_std = eeg_np.std(axis=1, keepdims=True) + 1e-12
            e_norm = (eeg_np - e_mean) / e_std
            
            a_mean = a_np.mean(axis=1, keepdims=True)
            a_std = a_np.std(axis=1, keepdims=True) + 1e-12
            a_norm = (a_np - a_mean) / a_std
            
            time_steps = e_norm.shape[1]
            lagged_blocks = []
            for lag in range(num_lags):
                if lag == 0:
                    lagged_blocks.append(e_norm.T)
                else:
                    shifted = np.vstack([np.zeros((lag, num_channels)), e_norm.T[:-lag]])
                    lagged_blocks.append(shifted)
                    
            X_mat = np.concatenate(lagged_blocks, axis=1)
            Y_mat = a_norm.T
            
            s_xtx += X_mat.T @ X_mat
            s_xty += X_mat.T @ Y_mat
            
        subject_xtx[sub] = s_xtx
        subject_xty[sub] = s_xty
        global_xtx += s_xtx
        global_xty += s_xty
        
    print("\nPhase 2: Evaluating Alternative Similarity Metrics...")
    ridge_lambda = 100.0
    
    metrics_list = ["Pearson", "Cosine", "Lagged_Corr", "CCA", "Mutual_Info"]
    
    results = {
        metric: {"trials_ok": 0, "total_trials": 0, "win_ok": 0, "win_tot": 0} 
        for metric in metrics_list
    }
    
    win_samples = int(10 * 64)
    hop_samples = int(1.0 * 64)
    
    start_time = time.time()
    
    for held_out_sub in all_subject_data.keys():
        print(f"  Evaluating held-out Subject {held_out_sub}...")
        train_xtx = global_xtx - subject_xtx[held_out_sub]
        train_xty = global_xty - subject_xty[held_out_sub]
        
        regularized = train_xtx + ridge_lambda * np.eye(feature_count, dtype=float)
        W = np.linalg.solve(regularized, train_xty)
        
        for t in all_subject_data[held_out_sub]:
            eeg_np = t["eeg"].numpy()
            audio_a = t["audio_a"].numpy()
            audio_b = t["audio_b"].numpy()
            
            e_mean = eeg_np.mean(axis=1, keepdims=True)
            e_std = eeg_np.std(axis=1, keepdims=True) + 1e-12
            e_norm = (eeg_np - e_mean) / e_std
            
            a_mean = audio_a.mean(axis=1, keepdims=True)
            a_std = audio_a.std(axis=1, keepdims=True) + 1e-12
            a_norm = (audio_a - a_mean) / a_std
            
            b_mean = audio_b.mean(axis=1, keepdims=True)
            b_std = audio_b.std(axis=1, keepdims=True) + 1e-12
            b_norm = (audio_b - b_mean) / b_std
            
            lagged_blocks = []
            for lag in range(num_lags):
                if lag == 0:
                    lagged_blocks.append(e_norm.T)
                else:
                    shifted = np.vstack([np.zeros((lag, num_channels)), e_norm.T[:-lag]])
                    lagged_blocks.append(shifted)
            X_mat = np.concatenate(lagged_blocks, axis=1)
            
            pred = (X_mat @ W).T # (28, Time)
            
            # Evaluate across windows
            trial_correct = {m: 0 for m in metrics_list}
            trial_total_windows = 0
            
            if win_samples >= pred.shape[1]:
                sims = evaluate_window(pred, a_norm, b_norm)
                for m in metrics_list:
                    a_score, b_score = sims[m]
                    if a_score > b_score:
                        results[m]["win_ok"] += 1
                        trial_correct[m] += 1
                    results[m]["win_tot"] += 1
                trial_total_windows = 1
            else:
                for start in range(0, pred.shape[1] - win_samples + 1, hop_samples):
                    stop = start + win_samples
                    
                    p_win = pred[:, start:stop]
                    a_win = a_norm[:, start:stop]
                    b_win = b_norm[:, start:stop]
                    
                    sims = evaluate_window(p_win, a_win, b_win)
                    
                    for m in metrics_list:
                        a_score, b_score = sims[m]
                        if a_score > b_score:
                            results[m]["win_ok"] += 1
                            trial_correct[m] += 1
                        results[m]["win_tot"] += 1
                        
                    trial_total_windows += 1
                    
            for m in metrics_list:
                if trial_correct[m] > trial_total_windows / 2.0:
                    results[m]["trials_ok"] += 1
                results[m]["total_trials"] += 1

    elapsed = time.time() - start_time
    
    print("\n================================================================")
    print("FINAL RESULTS: COMPARISON OF SIMILARITY METRICS")
    print("================================================================")
    print(f"{'Metric':<15} | {'Trial Accuracy':<15} | {'Window Accuracy'}")
    print("-" * 55)
    
    for m in metrics_list:
        t_acc = results[m]["trials_ok"] / results[m]["total_trials"]
        w_acc = results[m]["win_ok"] / results[m]["win_tot"]
        print(f"{m:<15} | {t_acc*100:>14.2f}% | {w_acc*100:>14.2f}%")
        
    print(f"\nEvaluation took {elapsed:.1f} seconds.")

if __name__ == "__main__":
    main()
