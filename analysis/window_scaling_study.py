import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from preprocessing.dataset import get_mapping_data, subject_files, load_subject_examples
from training.train_matchnet_loso import prepare_dataset, chunk_trial, evaluate_model
from models.matchnet import ContrastiveMatchNet

FS = 64
CHANNELS = [13, 46, 43, 23, 50, 0, 52, 14]
LOWCUT = 1.0
HIGHCUT = 6.0
NUM_BANDS = 28

def run_scaling_study(target_subjects=["S8", "S9", "S11"], windows=[0.5, 1, 2, 3, 5, 7, 10, 20, 30, 60]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Window Scaling Study on {device}")
    
    mapping, envelopes = get_mapping_data()
    all_paths = subject_files()
    if not all_paths:
        print("No subjects found.")
        return
        
    subject_examples = {str(p): load_subject_examples(p) for p in all_paths}
    
    results = []
    
    for held_out_subj in target_subjects:
        held_out_path = next((p for p in all_paths if p.stem.split('_')[0] == held_out_subj), None)
        if not held_out_path:
            print(f"Warning: Subject {held_out_subj} not found, skipping.")
            continue
            
        print(f"\n--- Processing Subject {held_out_subj} ---")
        
        # Load best checkpoint
        checkpoint_path = Path("checkpoints") / f"matchnet_fold_{held_out_path.stem}_best.pth"
        if not checkpoint_path.exists():
            print(f"Warning: Checkpoint {checkpoint_path} not found. Skipping.")
            continue
            
        model = ContrastiveMatchNet(eeg_model_type="eegnet", eeg_channels=len(CHANNELS), audio_channels=NUM_BANDS, latent_dim=64).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        
        test_exs = subject_examples[str(held_out_path)]
        X_te_full, YA_te_full, YB_te_full = prepare_dataset(test_exs, CHANNELS, LOWCUT, HIGHCUT, held_out_path.stem, mapping, envelopes)
        
        for w_sec in windows:
            nc, nt = evaluate_model(model, X_te_full, YA_te_full, YB_te_full, device, window_sec=w_sec, metric="pearson")
            acc = nc / max(nt, 1)
            
            results.append({
                "subject": held_out_subj,
                "window_sec": w_sec,
                "accuracy": acc * 100.0,
                "num_correct": nc,
                "num_trials": nt
            })
            print(f"  Window {w_sec:>4}s | Acc: {acc*100:5.2f}% ({nc}/{nt})")

    if not results:
        print("No results generated.")
        return
        
    # Create Dataframe
    df = pd.DataFrame(results)
    
    # Save CSV
    os.makedirs("results/statistics", exist_ok=True)
    csv_path = "results/statistics/window_scaling_study.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV to {csv_path}")
    
    # Interpretations & Markdown Generation
    os.makedirs("results/reports", exist_ok=True)
    md_path = "results/reports/window_scaling_study.md"
    
    with open(md_path, "w") as f:
        f.write("# Window Scaling Study Report\n\n")
        f.write("## Overview\n")
        f.write("This report evaluates baseline MatchNet accuracy as the decision window scales from 0.5s to 60s without retraining.\n\n")
        
        f.write("## Results by Subject\n\n")
        for subj in target_subjects:
            subj_df = df[df["subject"] == subj]
            if subj_df.empty:
                continue
                
            f.write(f"### Subject {subj}\n\n")
            f.write("| Window | Accuracy | Correct | Total |\n")
            f.write("| ------ | -------: | ------: | ----: |\n")
            
            acc_05 = 0.0
            acc_60 = 0.0
            
            for _, row in subj_df.iterrows():
                f.write(f"| {row['window_sec']:>4}s   | {row['accuracy']:>7.2f}% | {row['num_correct']:>7} | {row['num_trials']:>5} |\n")
                if row['window_sec'] == 0.5: acc_05 = row['accuracy']
                if row['window_sec'] == 60: acc_60 = row['accuracy']
                
            improvement = acc_60 - acc_05
            f.write(f"\n**Interpretation:**\n")
            f.write(f"- 0.5s Accuracy: {acc_05:.2f}%\n")
            f.write(f"- 60s Accuracy: {acc_60:.2f}%\n")
            f.write(f"- Absolute Gain: {improvement:+.2f}%\n\n")
            
            if improvement > 15.0:
                f.write("> **Strongly Improves**: The model successfully accumulates evidence over time, indicating short-window performance is primarily limited by the short-window architecture's ability to extract features, rather than an underlying lack of neural signal.\n\n")
            elif improvement >= 5.0:
                f.write("> **Moderately Improves**: The model extracts some additional evidence over time, but faces a ceiling.\n\n")
            else:
                f.write("> **Saturates**: Performance flatlines. The current features/objectives are already extracting all available information from this subject very early.\n\n")
                
    print(f"Saved Markdown Report to {md_path}")
    
    # Generate Plot
    os.makedirs("results/plots", exist_ok=True)
    plot_path = "results/plots/window_scaling_study.png"
    
    plt.figure(figsize=(10, 6))
    for subj in target_subjects:
        subj_df = df[df["subject"] == subj]
        if not subj_df.empty:
            plt.plot(subj_df["window_sec"], subj_df["accuracy"], marker='o', linewidth=2, label=subj)
            
    plt.xscale('log')
    plt.xticks(windows, labels=[str(w) for w in windows])
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.title("Evidence Accumulation: Accuracy vs Decision Window Length")
    plt.xlabel("Window Length (seconds, log scale)")
    plt.ylabel("Accuracy (%)")
    plt.ylim(40, 100)
    plt.legend(title="Subject")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Saved Plot to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Evidence Accumulation Scaling Study")
    parser.add_argument("--subjects", nargs='+', default=["S8", "S9", "S11"], help="Subjects to evaluate")
    args = parser.parse_args()
    
    # We must enforce the specific window ranges requested
    run_scaling_study(target_subjects=args.subjects)
