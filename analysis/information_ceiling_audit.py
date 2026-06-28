import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.kul_cached_dataset import KULCachedLoader

def safe_corr_np(x, y, eps=1e-8):
    x_mean = x.mean()
    y_mean = y.mean()
    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
    return num / (den + eps)

def extract_story_name(stimulus_filename):
    """Extracts story ID from filename, e.g., 'part1_track1_dry.wav' -> 'part1'."""
    name = stimulus_filename.lower()
    if "part1" in name: return "Part 1"
    if "part2" in name: return "Part 2"
    if "part3" in name: return "Part 3"
    if "part4" in name: return "Part 4"
    return "Unknown"

def evaluate_trial(pred, wav_a, wav_b, window_sec=10, hop_sec=1.0, fs=64):
    num_bands = pred.shape[0]
    win_samples = int(window_sec * fs)
    hop_samples = int(hop_sec * fs)
    
    if win_samples >= pred.shape[1]:
        c_a = np.mean([safe_corr_np(pred[i], wav_a[i]) for i in range(num_bands)])
        c_b = np.mean([safe_corr_np(pred[i], wav_b[i]) for i in range(num_bands)])
        return c_a > c_b, 1, 1 if c_a > c_b else 0
        
    correct_windows = 0
    total_windows = 0
    
    for start in range(0, pred.shape[1] - win_samples + 1, hop_samples):
        stop = start + win_samples
        c_a = np.mean([safe_corr_np(pred[i, start:stop], wav_a[i, start:stop]) for i in range(num_bands)])
        c_b = np.mean([safe_corr_np(pred[i, start:stop], wav_b[i, start:stop]) for i in range(num_bands)])
        if c_a > c_b:
            correct_windows += 1
        total_windows += 1
        
    if total_windows == 0:
        return False, 0, 0
        
    trial_correct = (correct_windows > total_windows / 2.0)
    return trial_correct, total_windows, correct_windows

def main():
    print("================================================================")
    print("             INFORMATION CEILING AUDIT (KUL DATASET)            ")
    print("================================================================")
    
    loader = KULCachedLoader(REPO_ROOT / "data" / "processed_kul")
    try:
        all_subject_data = loader.load_all()
    except FileNotFoundError:
        print("KUL Cache not found. Please run preprocessing first.")
        return
        
    out_dir = REPO_ROOT / "results" / "ceiling_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    num_channels = 8
    num_lags = 17 # 0 to 16 inclusive (approx 250ms at 64Hz)
    feature_count = num_channels * num_lags
    num_bands = 28
    
    subject_xtx = {}
    subject_xty = {}
    
    global_xtx = np.zeros((feature_count, feature_count), dtype=float)
    global_xty = np.zeros((feature_count, num_bands), dtype=float)
    
    print("\nPhase 1: Accumulating Global Ridge Matrices...")
    # Upper Bound Heatmap Variables
    eeg_audio_cross_corr = np.zeros((num_channels, num_lags))
    total_cross_corr_trials = 0
    
    for sub, trials in all_subject_data.items():
        s_xtx = np.zeros((feature_count, feature_count), dtype=float)
        s_xty = np.zeros((feature_count, num_bands), dtype=float)
        
        for t in trials:
            eeg_np = t["eeg"].numpy()
            a_np = t["audio_a"].numpy()
            
            # Normalization
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
                    
            X_mat = np.concatenate(lagged_blocks, axis=1) # (Time, Channels*Lags)
            Y_mat = a_norm.T # (Time, 28)
            
            s_xtx += X_mat.T @ X_mat
            s_xty += X_mat.T @ Y_mat
            
            # Upper Bound Heatmap Logic
            # Correlate raw EEG (per channel, per lag) directly with target envelope (averaged across bands)
            target_1d = a_norm.mean(axis=0) # Average across 28 bands for a single 1D target
            for lag in range(num_lags):
                for ch in range(num_channels):
                    feature = X_mat[:, lag * num_channels + ch]
                    c = safe_corr_np(feature, target_1d)
                    eeg_audio_cross_corr[ch, lag] += c
            total_cross_corr_trials += 1
            
        subject_xtx[sub] = s_xtx
        subject_xty[sub] = s_xty
        
        global_xtx += s_xtx
        global_xty += s_xty
        
    eeg_audio_cross_corr /= total_cross_corr_trials
    
    print("\nPhase 2: Fast LOSO Ridge Evaluation & Metrics Collection...")
    ridge_lambda = 100.0
    
    subject_metrics = defaultdict(lambda: {"trials_ok": 0, "win_ok": 0, "win_tot": 0, "margins": []})
    story_metrics = defaultdict(lambda: {"trials_ok": 0, "win_ok": 0, "win_tot": 0, "margins": []})
    
    all_margins = []
    residual_variances = []
    residual_autocorrs = []
    
    # SNR tracking per subject
    subject_snr = defaultdict(list)
    subject_variance = defaultdict(list)
    
    for held_out_sub in all_subject_data.keys():
        train_xtx = global_xtx - subject_xtx[held_out_sub]
        train_xty = global_xty - subject_xty[held_out_sub]
        
        regularized = train_xtx + ridge_lambda * np.eye(feature_count, dtype=float)
        W = np.linalg.solve(regularized, train_xty) # (Channels*Lags, 28)
        
        for t in all_subject_data[held_out_sub]:
            story = extract_story_name(t["meta"]["stimuli_left"])
            
            eeg_np = t["eeg"].numpy()
            audio_a = t["audio_a"].numpy()
            audio_b = t["audio_b"].numpy()
            
            # Signal Quality (Exp 6)
            # Since KUL is bandpassed 1-8Hz during preprocessing, Alpha/Beta/Gamma are 0.
            # We measure raw variance and a pseudo-SNR (signal power / noise approximation)
            sig_var = np.var(eeg_np)
            subject_variance[held_out_sub].append(sig_var)
            
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
            
            c_att = np.mean([safe_corr_np(pred[i], a_norm[i]) for i in range(num_bands)])
            c_unatt = np.mean([safe_corr_np(pred[i], b_norm[i]) for i in range(num_bands)])
            margin = c_att - c_unatt
            
            trial_ok, n_win, c_win = evaluate_trial(pred, a_norm, b_norm, window_sec=10, hop_sec=1.0)
            
            # Update metrics
            subject_metrics[held_out_sub]["trials_ok"] += int(trial_ok)
            subject_metrics[held_out_sub]["win_ok"] += c_win
            subject_metrics[held_out_sub]["win_tot"] += n_win
            subject_metrics[held_out_sub]["margins"].append(margin)
            
            story_metrics[story]["trials_ok"] += int(trial_ok)
            story_metrics[story]["win_ok"] += c_win
            story_metrics[story]["win_tot"] += n_win
            story_metrics[story]["margins"].append(margin)
            
            all_margins.append(margin)
            
            # Residual Structure (Exp 4)
            residual = a_norm - pred
            residual_variances.append(np.var(residual))
            
            # Autocorrelation at 100ms lag (~6 samples at 64Hz)
            if residual.shape[1] > 6:
                autocorr_100ms = np.mean([safe_corr_np(residual[i, 6:], residual[i, :-6]) for i in range(num_bands)])
                residual_autocorrs.append(autocorr_100ms)
                
    print("\n================================================================")
    print("EXPERIMENT 1 & 6: Subject-wise Performance & Signal Quality")
    print("================================================================")
    print(f"{'Subject':<10} | {'Trial Acc':<12} | {'Win Acc':<10} | {'Med Margin':<12} | {'EEG Variance'}")
    print("-" * 65)
    
    sub_accs = []
    for sub in sorted(subject_metrics.keys(), key=lambda x: int(x[1:])):
        m = subject_metrics[sub]
        t_acc = m["trials_ok"] / len(m["margins"])
        w_acc = m["win_ok"] / m["win_tot"]
        med_marg = np.median(m["margins"])
        var = np.mean(subject_variance[sub])
        
        sub_accs.append(t_acc)
        print(f"{sub:<10} | {t_acc*100:>8.1f}%   | {w_acc*100:>7.1f}% | {med_marg:>10.4f} | {var:>10.4f}")
        
    print(f"Overall Subject Median Trial Acc: {np.median(sub_accs)*100:.1f}%\n")
    
    print("\n================================================================")
    print("EXPERIMENT 2: Story-wise Performance")
    print("================================================================")
    print(f"{'Story':<10} | {'Trial Acc':<12} | {'Win Acc':<10} | {'Med Margin':<12} | {'Total Trials'}")
    print("-" * 65)
    
    for story in sorted(story_metrics.keys()):
        m = story_metrics[story]
        if len(m["margins"]) > 0:
            t_acc = m["trials_ok"] / len(m["margins"])
            w_acc = m["win_ok"] / m["win_tot"]
            med_marg = np.median(m["margins"])
            print(f"{story:<10} | {t_acc*100:>8.1f}%   | {w_acc*100:>7.1f}% | {med_marg:>10.4f} | {len(m['margins'])}")
            
    print("\n================================================================")
    print("EXPERIMENT 4: Residual Structure")
    print("================================================================")
    print(f"Mean Residual Variance:      {np.mean(residual_variances):.4f}")
    print(f"Mean Autocorrelation (100ms): {np.mean(residual_autocorrs):.4f}")
    
    print("\n================================================================")
    print("Generating Plots...")
    print("================================================================")
    
    # 1. Subject Accuracies
    plt.figure(figsize=(10, 5))
    subs = sorted(subject_metrics.keys(), key=lambda x: int(x[1:]))
    accs = [subject_metrics[s]["trials_ok"] / len(subject_metrics[s]["margins"]) * 100 for s in subs]
    sns.barplot(x=subs, y=accs, color='royalblue')
    plt.axhline(y=50, color='r', linestyle='--', label='Chance (50%)')
    plt.axhline(y=65, color='g', linestyle='--', label='Empirical Ceiling (65%)')
    plt.title("Leave-One-Subject-Out (LOSO) Trial Accuracy by Subject")
    plt.ylabel("Trial Accuracy (%)")
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "subject_accuracies.png")
    plt.close()
    
    # 2. Margin Histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(all_margins, bins=50, kde=True, color='purple')
    plt.axvline(x=0, color='r', linestyle='--', label='Zero Margin')
    plt.title("Distribution of Trial Margins (Corr_Att - Corr_Unatt)")
    plt.xlabel("Correlation Margin")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "margin_histogram.png")
    plt.close()
    
    # 3. Upper Bound Correlation Heatmap (EEG Channels vs Target)
    plt.figure(figsize=(10, 6))
    sns.heatmap(eeg_audio_cross_corr, cmap="coolwarm", center=0, cbar_kws={'label': 'Pearson Correlation'})
    plt.title("Upper Bound: Raw EEG vs Target Envelope Correlation")
    plt.xlabel("Temporal Lag (Samples at 64Hz, 1 sample ≈ 15.6ms)")
    plt.ylabel("EEG Channel Index")
    plt.tight_layout()
    plt.savefig(out_dir / "eeg_envelope_correlation_heatmap.png")
    plt.close()
    
    print(f"Audit complete! Plots saved to {out_dir}")

if __name__ == "__main__":
    main()
