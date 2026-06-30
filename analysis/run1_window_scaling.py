import os
import sys
import torch
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.kul_cached_dataset import KULCachedLoader
from models.aad_conformer import AADConformer

RESULTS_DIR = REPO_ROOT / "results" / "conformer_loso"
SEEDS = [1, 7, 21, 42, 123]
WINDOWS = [1, 2, 5, 10, 20, 30, 60]

def safe_corr_np(x, y, eps=1e-8):
    x_mean = x.mean()
    y_mean = y.mean()
    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
    return num / (den + eps)

def evaluate_trial_majority_vote(predicted: np.ndarray, wav_a: np.ndarray, wav_b: np.ndarray, window_seconds: int, hop_seconds: float = 1.0, fs: int = 64):
    win_samples = int(window_seconds * fs)
    hop_samples = int(hop_seconds * fs)
    
    if win_samples >= predicted.shape[0]:
        c_a = safe_corr_np(predicted, wav_a)
        c_b = safe_corr_np(predicted, wav_b)
        return c_a > c_b, 1, 1 if c_a > c_b else 0
        
    correct_windows = 0
    total_windows = 0
    
    for start in range(0, predicted.shape[0] - win_samples + 1, hop_samples):
        stop = start + win_samples
        c_a = safe_corr_np(predicted[start:stop], wav_a[start:stop])
        c_b = safe_corr_np(predicted[start:stop], wav_b[start:stop])
        if c_a > c_b:
            correct_windows += 1
        total_windows += 1
        
    if total_windows == 0:
        return False, 0, 0
        
    trial_correct = (correct_windows > total_windows / 2.0)
    return trial_correct, total_windows, correct_windows

def main():
    if Path("/kaggle/input/datasets/lowk1ee/kul-preprocessed-cache/data/processed_kul").exists():
        cache_dir = Path("/kaggle/input/datasets/lowk1ee/kul-preprocessed-cache/data/processed_kul")
    elif Path("/kaggle/input/kul-preprocessed-cache/data/processed_kul").exists():
        cache_dir = Path("/kaggle/input/kul-preprocessed-cache/data/processed_kul")
    else:
        cache_dir = REPO_ROOT / "data" / "processed_kul"
        
    try:
        loader = KULCachedLoader(cache_dir)
        all_subject_data = loader.load_all()
    except FileNotFoundError:
        print(f"KUL Cache not found at {cache_dir}. Cannot run window scaling.")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    subjects = sorted(list(all_subject_data.keys()))
    
    window_results = {w: [] for w in WINDOWS}
    records = []
    
    for seed in SEEDS:
        print(f"\n--- Processing Seed {seed} ---")
        checkpoint_dir = RESULTS_DIR / "checkpoints" / f"seed_{seed}"
        
        if not checkpoint_dir.exists():
            print(f"Checkpoints for seed {seed} not found at {checkpoint_dir}. Skipping.")
            continue
            
        seed_window_accs = {w: [] for w in WINDOWS}
        
        for test_subject in subjects:
            ckpt_path = checkpoint_dir / f"model_{test_subject}.pt"
            if not ckpt_path.exists():
                print(f"Warning: {ckpt_path} missing.")
                continue
                
            model = AADConformer(
                in_channels=8, temporal_filters=32, spatial_filters=64,
                embed_dim=64, num_heads=4, num_layers=2, dropout=0.3, stride=4
            ).to(device)
            
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            model.eval()
            
            test_trials = all_subject_data[test_subject]
            
            # For each trial, run the model ONCE, then evaluate across all windows
            subject_window_correct = {w: 0 for w in WINDOWS}
            total_trials = len(test_trials)
            
            with torch.no_grad():
                for t in test_trials:
                    eeg = t["eeg"].unsqueeze(0).to(device)       
                    audio_a = t["audio_a"].unsqueeze(0).to(device) 
                    audio_b = t["audio_b"].unsqueeze(0).to(device)
                    
                    audio_a = audio_a.mean(dim=1, keepdim=True)
                    audio_b = audio_b.mean(dim=1, keepdim=True)
                    
                    eeg_mean = eeg.mean(dim=2, keepdim=True)
                    eeg_std = eeg.std(dim=2, keepdim=True) + 1e-8
                    eeg_norm = (eeg - eeg_mean) / eeg_std
                    
                    audio_a_mean = audio_a.mean(dim=2, keepdim=True)
                    audio_a_std = audio_a.std(dim=2, keepdim=True) + 1e-8
                    audio_a_norm = (audio_a - audio_a_mean) / audio_a_std
                    
                    audio_b_mean = audio_b.mean(dim=2, keepdim=True)
                    audio_b_std = audio_b.std(dim=2, keepdim=True) + 1e-8
                    audio_b_norm = (audio_b - audio_b_mean) / audio_b_std
                    
                    pred = model(eeg_norm)
                    
                    pred_np = pred.squeeze(0).cpu().numpy()
                    wav_a_np = audio_a_norm.squeeze(1).squeeze(0).cpu().numpy()
                    wav_b_np = audio_b_norm.squeeze(1).squeeze(0).cpu().numpy()
                    
                    for w in WINDOWS:
                        trial_ok, n_win, c_win = evaluate_trial_majority_vote(pred_np, wav_a_np, wav_b_np, window_seconds=w, hop_seconds=1.0, fs=64)
                        if trial_ok:
                            subject_window_correct[w] += 1
                            
            for w in WINDOWS:
                acc = subject_window_correct[w] / total_trials if total_trials > 0 else 0
                seed_window_accs[w].append(acc)
                
        # Aggregate seed level
        for w in WINDOWS:
            mean_acc = np.mean(seed_window_accs[w])
            window_results[w].append(mean_acc)
            print(f"  Window {w}s Accuracy: {mean_acc*100:.1f}%")
            
    # Final aggregation across seeds
    print("\n=== FINAL WINDOW SCALING ===")
    plot_means = []
    plot_stds = []
    
    for w in WINDOWS:
        w_accs = window_results[w]
        if not w_accs:
            continue
        w_mean = np.mean(w_accs)
        w_std = np.std(w_accs)
        plot_means.append(w_mean)
        plot_stds.append(w_std)
        print(f"{w:2d}s Decision Window: {w_mean*100:.2f}% ± {w_std*100:.2f}%")
        records.append({"Window (s)": w, "Mean Accuracy (%)": w_mean*100, "Std (%)": w_std*100})
        
    # Generate Figure
    if plot_means:
        plt.figure(figsize=(8, 6))
        plt.errorbar(WINDOWS, np.array(plot_means)*100, yerr=np.array(plot_stds)*100, marker='o', capsize=5, linestyle='-', color='b')
        plt.xlabel("Decision Window Length (s)")
        plt.ylabel("LOSO Trial Accuracy (%)")
        plt.title("Conformer Scaling: Accuracy vs. Decision Window Length")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.xticks(WINDOWS)
        
        out_dir = RESULTS_DIR / "figures"
        out_dir.mkdir(exist_ok=True, parents=True)
        
        plt.savefig(out_dir / "window_scaling_curve.png", dpi=300)
        plt.close()
        
        df = pd.DataFrame(records)
        df.to_csv(RESULTS_DIR / "window_scaling_table.csv", index=False)
        print(f"\nSaved scaling plots and tables to {RESULTS_DIR}")

if __name__ == "__main__":
    main()
