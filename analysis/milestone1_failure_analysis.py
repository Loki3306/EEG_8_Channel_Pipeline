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
                abs_margin = abs(margin)
                is_correct = 1 if margin > 0 else 0
                
                trial_stats.append({
                    "trial_idx": trial_idx,
                    "segment_idx": seg_idx,
                    "sim_a": sim_a,
                    "sim_b": sim_b,
                    "margin": margin,
                    "abs_margin": abs_margin,
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

def generate_failure_diagnosis(row):
    """
    Generates an evidence-driven string explaining why a subject failed.
    """
    reasons = []
    
    att_corr = row['mean_attended_corr']
    unatt_corr = row['mean_unattended_corr']
    med_margin = row['median_margin']
    med_abs_margin = row['median_abs_margin']
    acc = row['accuracy']
    
    if att_corr < 0.02:
        reasons.append(f"Weak Signal Quality (Attended Corr = {att_corr:.3f} is extremely low). The model fails to extract meaningful task-related features.")
    elif unatt_corr > att_corr * 0.8:
        reasons.append(f"Discrimination Failure (Unattended Corr {unatt_corr:.3f} is too close to Attended Corr {att_corr:.3f}). The model extracts audio but cannot separate streams.")
    
    if med_abs_margin < 0.02:
        reasons.append(f"High Uncertainty (Median Abs Margin = {med_abs_margin:.3f}). The model is basically guessing randomly across all trials.")
    elif med_margin < 0 and med_abs_margin > 0.05:
        reasons.append(f"Confident but Wrong (Median Margin = {med_margin:.3f}, Median Abs Margin = {med_abs_margin:.3f}). Representation mismatch. The model confidently predicts the wrong class.")
        
    if len(reasons) == 0:
        reasons.append(f"Marginal multi-factor degradation. Att Corr: {att_corr:.3f}, Unatt Corr: {unatt_corr:.3f}.")
        
    return " | ".join(reasons)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --------------------------------------------------------------------------
    # SCIENTIFIC AUDIT WARNING
    # This script replicates the Kaggle LOSO evaluation metric exactly, which means 
    # it uses wavA (Y_A) as the ground-truth target regardless of the actual label.
    # We are generating these metrics strictly to explain the published (but flawed) LOSO results.
    # --------------------------------------------------------------------------
    print("WARNING: This script replicates the exact Kaggle LOSO metric (which forces wavA as target).")
    
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
    
    for p in tqdm(all_paths, desc="Processing Subjects"):
        subject_id = p.stem
        ckpt_path = Path(f"checkpoints/matchnet_fold_{subject_id}_best.pth")
        
        if not ckpt_path.exists():
            continue
            
        model = MatchNet(in_channels=len(CHANNELS)).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        
        test_exs = subject_examples[str(p)]
        X_te, YA_te, YB_te = prepare_dataset(test_exs, CHANNELS, LOWCUT, HIGHCUT, subject_id, mapping, envelopes)
        
        t_stats = extract_trial_stats(model, X_te, YA_te, YB_te, device, window_sec=10)
        
        if len(t_stats) == 0:
            continue
            
        df_stats = pd.DataFrame(t_stats)
        
        # Calculate diagnostics
        att_corr = df_stats["sim_a"].mean()
        unatt_corr = df_stats["sim_b"].mean()
        margins = df_stats["margin"]
        abs_margins = df_stats["abs_margin"]
        
        weak_signal = 1.0 / (abs(att_corr) + 1e-4) if att_corr < 0.05 else 0.0
        discrim_fail = unatt_corr / (att_corr + 1e-4) if att_corr > 0 else 1.0
        uncertainty = 1.0 / (abs_margins.median() + 1e-4)
        
        all_subject_stats.append({
            "subject": subject_id.replace("_data_preproc", ""),
            "accuracy": df_stats["is_correct"].mean(),
            "mean_attended_corr": att_corr,
            "mean_unattended_corr": unatt_corr,
            "mean_margin": margins.mean(),
            "median_margin": margins.median(),
            "std_margin": margins.std(),
            "mean_abs_margin": abs_margins.mean(),
            "median_abs_margin": abs_margins.median(),
            "margin_25": np.percentile(margins, 25),
            "margin_50": np.percentile(margins, 50),
            "margin_75": np.percentile(margins, 75),
            "weak_signal_score": weak_signal,
            "discrimination_failure_score": discrim_fail,
            "uncertainty_score": uncertainty,
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
            
            # Simple render
            display_cols = ['subject', 'accuracy', 'mean_attended_corr', 'mean_unattended_corr', 'mean_margin', 'median_abs_margin']
            
            f.write("## 1. Worst Performing Subjects (Bottom Quartile)\n")
            bottom_q = df_summary.head(max(len(df_summary)//4, 4))
            f.write(bottom_q[display_cols].to_markdown(index=False) + "\n\n")
            
            f.write("## 2. Best Performing Subjects (Top Quartile)\n")
            top_q = df_summary.tail(max(len(df_summary)//4, 4))
            f.write(top_q[display_cols].to_markdown(index=False) + "\n\n")
            
            f.write("## 3. Why does this subject fail? (Bottom Quartile Diagnostics)\n")
            for _, row in bottom_q.iterrows():
                f.write(f"### {row['subject']} (Acc: {row['accuracy']*100:.1f}%)\n")
                f.write(f"**Diagnosis**: {generate_failure_diagnosis(row)}\n\n")
                f.write(f"- Margin: 25th: {row['margin_25']:.3f} | 50th: {row['margin_50']:.3f} | 75th: {row['margin_75']:.3f}\n")
                f.write(f"- Abs Margin (Median): {row['median_abs_margin']:.3f}\n")
                f.write(f"- Signal Score: {row['weak_signal_score']:.2f} | Discrim Score: {row['discrimination_failure_score']:.2f} | Uncertainty Score: {row['uncertainty_score']:.2f}\n\n")
    
    # --------------------------------------------------------------------------
    # Task 2: Window-Length Sensitivity (Worst 5, Median 5, Best 5)
    # --------------------------------------------------------------------------
    if len(df_summary) >= 15:
        worst_5 = df_summary.head(5)['subject'].tolist()
        best_5 = df_summary.tail(5)['subject'].tolist()
        median_idx = len(df_summary) // 2
        median_5 = df_summary.iloc[median_idx-2:median_idx+3]['subject'].tolist()
    else:
        worst_5 = df_summary.head(len(df_summary)//3)['subject'].tolist()
        best_5 = df_summary.tail(len(df_summary)//3)['subject'].tolist()
        median_5 = [s for s in df_summary['subject'] if s not in worst_5 and s not in best_5][:5]
        
    target_subjects = worst_5 + median_5 + best_5
    # Keep track of category
    subject_category = {s: "Worst 5" for s in worst_5}
    subject_category.update({s: "Median 5" for s in median_5})
    subject_category.update({s: "Best 5" for s in best_5})
    
    window_lengths = [2, 5, 10, 20, 30]
    
    print("\n--- Task 2: Window-Length Sensitivity Analysis ---")
    window_results = []
    
    for subject_id in tqdm(target_subjects, desc="Window Analysis"):
        full_sub_id = f"{subject_id}_data_preproc"
        ckpt_path = Path(f"checkpoints/matchnet_fold_{full_sub_id}_best.pth")
        
        if not ckpt_path.exists():
            continue
            
        model = MatchNet(in_channels=len(CHANNELS)).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        
        p_match = next((p for p in all_paths if full_sub_id in p.stem), None)
        if not p_match:
            continue
            
        test_exs = subject_examples[str(p_match)]
        X_te, YA_te, YB_te = prepare_dataset(test_exs, CHANNELS, LOWCUT, HIGHCUT, full_sub_id, mapping, envelopes)
        
        for w in window_lengths:
            acc = evaluate_window_length(model, X_te, YA_te, YB_te, device, window_sec=w)
            window_results.append({
                "subject": subject_id,
                "category": subject_category[subject_id],
                "window_sec": w,
                "accuracy": acc
            })
            
    df_windows = pd.DataFrame(window_results)
    if len(df_windows) > 0:
        df_pivot = df_windows.pivot(index=["category", "subject"], columns="window_sec", values="accuracy")
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
