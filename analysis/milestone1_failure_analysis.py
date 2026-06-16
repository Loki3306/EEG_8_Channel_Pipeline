import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F

# Add project root to sys.path so we can import from training, models, dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.matchnet import MatchNet
from training.train_matchnet_loso import get_mapping_data, prepare_dataset, chunk_trial
from dataset.utils import subject_files, load_subject_examples

FS = 64
CHANNELS = [13, 46, 43, 23, 50, 0, 52, 14]
LOWCUT = 1.0
HIGHCUT = 32.0

def pearson_corr(x, y, dim=1):
    x_centered = x - x.mean(dim=dim, keepdim=True)
    y_centered = y - y.mean(dim=dim, keepdim=True)
    cov = (x_centered * y_centered).sum(dim=dim)
    var_x = (x_centered ** 2).sum(dim=dim)
    var_y = (y_centered ** 2).sum(dim=dim)
    return cov / torch.sqrt(var_x * var_y + 1e-8)

def extract_trial_stats(model, X, Y_A, Y_B, device, window_sec=10):
    model.eval()
    window_samples = int(window_sec * FS)
    
    trial_stats = []
    
    with torch.no_grad():
        for trial_idx in range(len(X)):
            x_np = X[trial_idx]
            ya_np = Y_A[trial_idx]
            yb_np = Y_B[trial_idx]
            
            start = 0
            seg_idx = 0
            while start + window_samples <= x_np.shape[1]:
                end = start + window_samples
                x_chunk = torch.FloatTensor(x_np[:, start:end]).unsqueeze(0).to(device)
                ya_chunk = torch.FloatTensor(ya_np[:, start:end]).unsqueeze(0).to(device)
                yb_chunk = torch.FloatTensor(yb_np[:, start:end]).unsqueeze(0).to(device)
                
                z_eeg, z_a, z_b = model(x_chunk, ya_chunk, yb_chunk)
                
                sim_a = pearson_corr(z_eeg, z_a, dim=1).mean().item()
                sim_b = pearson_corr(z_eeg, z_b, dim=1).mean().item()
                
                margin = sim_a - sim_b
                is_correct = 1 if margin > 0 else 0
                
                trial_stats.append({
                    "trial_idx": trial_idx,
                    "segment_idx": seg_idx,
                    "sim_a": sim_a,
                    "sim_b": sim_b,
                    "margin": margin,
                    "is_correct": is_correct
                })
                
                start += window_samples
                seg_idx += 1
                
    return trial_stats

def evaluate_window_length(model, X, Y_A, Y_B, device, window_sec):
    model.eval()
    window_samples = int(window_sec * FS)
    n_correct = 0.0
    n_total = 0
    
    with torch.no_grad():
        for trial_idx in range(len(X)):
            x_np = X[trial_idx]
            ya_np = Y_A[trial_idx]
            yb_np = Y_B[trial_idx]
            
            start = 0
            while start + window_samples <= x_np.shape[1]:
                end = start + window_samples
                x_chunk = torch.FloatTensor(x_np[:, start:end]).unsqueeze(0).to(device)
                ya_chunk = torch.FloatTensor(ya_np[:, start:end]).unsqueeze(0).to(device)
                yb_chunk = torch.FloatTensor(yb_np[:, start:end]).unsqueeze(0).to(device)
                
                z_eeg, z_a, z_b = model(x_chunk, ya_chunk, yb_chunk)
                
                sim_a = pearson_corr(z_eeg, z_a, dim=1).mean().item()
                sim_b = pearson_corr(z_eeg, z_b, dim=1).mean().item()
                
                if sim_a > sim_b:
                    n_correct += 1.0
                elif sim_a == sim_b:
                    n_correct += 0.5
                    
                n_total += 1
                start += window_samples
                
    return n_correct / max(n_total, 1)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    mapping, envelopes = get_mapping_data()
    all_paths = subject_files()
    
    if not all_paths:
        print("No subject data found. Exiting.")
        return
        
    subject_examples = {str(p): load_subject_examples(p) for p in all_paths}
    
    results_dir = Path("results/statistics")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # --------------------------------------------------------------------------
    # Task 1: Trial-Level Analysis (10s window) for all subjects
    # --------------------------------------------------------------------------
    print("\n--- Task 1: Trial-Level Subject Failure Analysis (10s) ---")
    all_subject_stats = []
    
    for p in all_paths:
        subject_id = p.stem
        ckpt_path = Path(f"checkpoints/matchnet_fold_{subject_id}_best.pth")
        
        if not ckpt_path.exists():
            print(f"  [Warning] Checkpoint not found for {subject_id}. Skipping.")
            continue
            
        print(f"  Evaluating {subject_id}...")
        
        # Load model
        model = MatchNet(in_channels=len(CHANNELS)).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        
        # Prepare held-out data
        test_exs = subject_examples[str(p)]
        X_te, YA_te, YB_te = prepare_dataset(test_exs, CHANNELS, LOWCUT, HIGHCUT, subject_id, mapping, envelopes)
        
        # Extract stats
        t_stats = extract_trial_stats(model, X_te, YA_te, YB_te, device, window_sec=10)
        
        if len(t_stats) == 0:
            continue
            
        df_stats = pd.DataFrame(t_stats)
        
        all_subject_stats.append({
            "subject": subject_id,
            "accuracy": df_stats["is_correct"].mean(),
            "mean_attended_corr": df_stats["sim_a"].mean(),
            "mean_unattended_corr": df_stats["sim_b"].mean(),
            "mean_margin": df_stats["margin"].mean(),
            "std_margin": df_stats["margin"].std(),
            "success_trials": df_stats["is_correct"].sum(),
            "failed_trials": len(df_stats) - df_stats["is_correct"].sum(),
            "total_trials": len(df_stats)
        })
        
    df_summary = pd.DataFrame(all_subject_stats)
    if len(df_summary) > 0:
        df_summary = df_summary.sort_values("accuracy", ascending=True)
        df_summary.to_csv(results_dir / "subject_failure_summary.csv", index=False)
        
        with open(results_dir / "subject_failure_report.md", "w") as f:
            f.write("# Subject Failure Analysis (10s Window)\n\n")
            f.write("This report answers whether poor subjects are failing due to weak neural signal (low attended correlation) or generalization breakdown (high unattended correlation).\n\n")
            f.write("## 1. Worst Performing Subjects (Bottom 5)\n")
            f.write(df_summary.head(5).to_markdown(index=False) + "\n\n")
            f.write("## 2. Best Performing Subjects (Top 5)\n")
            f.write(df_summary.tail(5).to_markdown(index=False) + "\n\n")
            f.write("## Summary of Findings\n")
            f.write("- **Low Attended Corr** implies poor signal quality or lack of attention.\n")
            f.write("- **High Unattended Corr** implies the model is extracting generalized audio features but failing to discriminate auditory streams.\n")
    
    # --------------------------------------------------------------------------
    # Task 2: Window-Length Sensitivity
    # --------------------------------------------------------------------------
    target_subjects = ["S11_data_preproc", "S6_data_preproc", "S8_data_preproc", "S7_data_preproc"]
    window_lengths = [2, 5, 10, 20, 30]
    
    print("\n--- Task 2: Window-Length Sensitivity Analysis ---")
    window_results = []
    
    for subject_id in target_subjects:
        ckpt_path = Path(f"checkpoints/matchnet_fold_{subject_id}_best.pth")
        
        # Attempt to map bare subject IDs like 'S11' to their preproc name if needed
        if not ckpt_path.exists():
            print(f"  [Warning] Checkpoint not found for {subject_id}. Skipping.")
            continue
            
        print(f"  Evaluating {subject_id} across windows...")
        
        model = MatchNet(in_channels=len(CHANNELS)).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        
        # Find matching path
        p_match = next((p for p in all_paths if subject_id in p.stem), None)
        if not p_match:
            continue
            
        test_exs = subject_examples[str(p_match)]
        X_te, YA_te, YB_te = prepare_dataset(test_exs, CHANNELS, LOWCUT, HIGHCUT, subject_id, mapping, envelopes)
        
        for w in window_lengths:
            acc = evaluate_window_length(model, X_te, YA_te, YB_te, device, window_sec=w)
            window_results.append({
                "subject": subject_id.split("_")[0],
                "window_sec": w,
                "accuracy": acc
            })
            
    df_windows = pd.DataFrame(window_results)
    if len(df_windows) > 0:
        # Pivot table for easier reading
        df_pivot = df_windows.pivot(index="subject", columns="window_sec", values="accuracy")
        df_windows.to_csv(results_dir / "window_length_analysis.csv", index=False)
        
        with open(results_dir / "window_length_analysis.md", "w") as f:
            f.write("# Window Length Sensitivity Analysis\n\n")
            f.write("Do poor subjects recover with longer evidence accumulation?\n\n")
            f.write(df_pivot.to_markdown() + "\n\n")
            f.write("## Decision Interpretation\n")
            f.write("- **Recovers Strongly**: Weak neural signal (implies calibration/cleaning needed).\n")
            f.write("- **Remains Poor / Saturates Early**: Generalization failure (implies representation learning failure).\n")

    print("\n[Done] Analysis saved to results/statistics/")

if __name__ == "__main__":
    main()
