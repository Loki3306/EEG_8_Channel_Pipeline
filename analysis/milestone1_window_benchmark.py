import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.matchnet import MatchNet
from training.train_matchnet_loso import get_mapping_data, prepare_dataset
from baselines.ridge_aad import subject_files, load_subject_examples

FS = 64
CHANNELS = [13, 46, 43, 23, 50, 0, 52, 14]
LOWCUT = 1.0
HIGHCUT = 6.0

def pearson_corr(x, y, dim=1):
    x_centered = x - x.mean(dim=dim, keepdim=True)
    y_centered = y - y.mean(dim=dim, keepdim=True)
    cov = (x_centered * y_centered).sum(dim=dim)
    var_x = (x_centered ** 2).sum(dim=dim)
    var_y = (y_centered ** 2).sum(dim=dim)
    return cov / torch.sqrt(var_x * var_y + 1e-8)

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
                
                # Y_A is guaranteed to be the attended stream due to the dataloader patch
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
        print("No subject data found. Ensure Kaggle inputs are mapped properly.")
        return
        
    subject_examples = {str(p): load_subject_examples(p) for p in all_paths}
    
    results_dir = Path("results/milestone1")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    window_lengths = [2, 5, 10, 20, 30]
    results = []
    
    print("\n--- Running Milestone 1: Window Length Benchmark ---")
    
    for p in tqdm(all_paths, desc="Processing Subjects"):
        subject_id = p.stem
        # Kaggle checkpoints are likely generated using the patched scripts
        ckpt_path = Path(f"checkpoints/matchnet_fold_{subject_id}_best.pth")
        
        if not ckpt_path.exists():
            print(f"Warning: Checkpoint not found for {subject_id}, skipping...")
            continue
            
        model = MatchNet(in_channels=len(CHANNELS)).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        
        test_exs = subject_examples[str(p)]
        X_te, YA_te, YB_te = prepare_dataset(test_exs, CHANNELS, LOWCUT, HIGHCUT, subject_id, mapping, envelopes)
        
        for w in window_lengths:
            acc = evaluate_window_length(model, X_te, YA_te, YB_te, device, window_sec=w)
            results.append({
                "subject": subject_id.replace("_data_preproc", ""),
                "window_sec": w,
                "accuracy": acc
            })
            
    df = pd.DataFrame(results)
    if len(df) == 0:
        print("No results generated. Ensure checkpoints exist.")
        return
        
    # 1. Subject-wise table
    df_pivot = df.pivot(index="subject", columns="window_sec", values="accuracy")
    df_pivot.to_csv(results_dir / "subject_window_accuracy.csv")
    
    with open(results_dir / "subject_window_accuracy.md", "w") as f:
        f.write("# Subject-wise Accuracy Across Windows\n\n")
        f.write(df_pivot.to_markdown() + "\n")
        
    # 2. Mean accuracy table
    mean_acc = df_pivot.mean()
    std_acc = df_pivot.std()
    
    df_mean = pd.DataFrame({"Mean Accuracy": mean_acc, "Std Dev": std_acc})
    df_mean.to_csv(results_dir / "mean_window_accuracy.csv")
    
    with open(results_dir / "mean_window_accuracy.md", "w") as f:
        f.write("# Mean Accuracy Across Windows\n\n")
        f.write(df_mean.to_markdown() + "\n")
        
    # 3. Generating the Accuracy vs Window Length figure
    plt.figure(figsize=(10, 6))
    plt.errorbar(mean_acc.index, mean_acc.values, yerr=std_acc.values, fmt='-o', capsize=5, label="Mean Accuracy")
    plt.axhline(y=0.5, color='r', linestyle='--', label="Chance Level")
    plt.title("Accuracy vs Window Length (Overall)")
    plt.xlabel("Window Length (seconds)")
    plt.ylabel("Decoding Accuracy")
    plt.ylim(0.4, 1.0)
    plt.xticks(window_lengths)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(results_dir / "accuracy_vs_window.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Recovery curves for worst subjects (based on 2s window performance)
    worst_subjects = df_pivot[2].nsmallest(5).index.tolist()
    
    plt.figure(figsize=(10, 6))
    for subj in worst_subjects:
        plt.plot(df_pivot.columns, df_pivot.loc[subj], marker='o', label=f"{subj}")
        
    plt.plot(df_pivot.columns, mean_acc.values, color='black', linewidth=3, linestyle='--', label="Population Mean")
    plt.axhline(y=0.5, color='r', linestyle='--', label="Chance Level")
    plt.title("Recovery Curves: Worst 5 Subjects")
    plt.xlabel("Window Length (seconds)")
    plt.ylabel("Decoding Accuracy")
    plt.ylim(0.4, 1.0)
    plt.xticks(window_lengths)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(results_dir / "worst_subjects_recovery.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[Done] Milestone 1 benchmark complete. Results saved to {results_dir}")

if __name__ == "__main__":
    main()
