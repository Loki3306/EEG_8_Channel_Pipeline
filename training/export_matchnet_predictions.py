"""
Phase 1: Confidence-Aware AAD Benchmarking
Step 1.1: Export raw MatchNet predictions and similarity scores.
"""

import os
import sys
import argparse
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.matchnet import ContrastiveMatchNet
from baselines.ridge_aad import subject_files, load_subject_examples
from training.train_matchnet_loso import prepare_dataset, get_mapping_data

def chunk_trial_with_metadata(x, ya, yb, subject_id, trial_id, label, window_sec, hop_sec, fs=64.0):
    window_samples = int(window_sec * fs)
    hop_samples = int(hop_sec * fs)
    
    n_chunks = max(0, (x.shape[1] - window_samples) // hop_samples + 1)
    
    chunks = []
    for i in range(n_chunks):
        start = i * hop_samples
        end = start + window_samples
        chunks.append({
            'subject_id': subject_id,
            'trial_id': trial_id,
            'window_id': i,
            'label': 'A' if label == 1 else 'B',
            'x': x[:, start:end],
            'ya': ya[:, start:end],
            'yb': yb[:, start:end]
        })
    return chunks

def pearson_corr(x, y, dim=1):
    x_centered = x - x.mean(dim=dim, keepdim=True)
    y_centered = y - y.mean(dim=dim, keepdim=True)
    cov = (x_centered * y_centered).sum(dim=dim)
    var_x = (x_centered ** 2).sum(dim=dim)
    var_y = (y_centered ** 2).sum(dim=dim)
    return cov / torch.sqrt(var_x * var_y + 1e-8)

def export_predictions(checkpoint_dir, out_csv, eeg_model="eegnet", channels=[13, 46, 43, 23, 50, 0, 52, 14], lowcut=1.0, highcut=6.0, window_sec=10.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Exporting predictions using device: {device}")
    
    mapping, envelopes = get_mapping_data()
    all_paths = subject_files()
    
    if not all_paths:
        print("No subjects found.")
        return
        
    csv_rows = []
    
    # Preload subjects to avoid repetitive disk access
    print("Loading datasets...")
    subject_examples = {str(p): load_subject_examples(p) for p in all_paths}
    
    for held_out_path in all_paths:
        subject_id = held_out_path.stem
        checkpoint_path = Path(checkpoint_dir) / f"matchnet_fold_{subject_id}_best.pth"
        
        if not checkpoint_path.exists():
            print(f"WARNING: Checkpoint {checkpoint_path} not found. Skipping subject {subject_id}.")
            continue
            
        print(f"Processing Subject {subject_id} using checkpoint {checkpoint_path.name}")
        
        model = ContrastiveMatchNet(eeg_model_type=eeg_model, eeg_channels=len(channels), audio_channels=28, latent_dim=64).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        
        test_exs = subject_examples[str(held_out_path)]
        
        with torch.no_grad():
            for trial_idx, ex in enumerate(test_exs):
                # Prepare single trial
                tX, tYA, tYB = prepare_dataset([ex], channels, lowcut, highcut, subject_id, mapping, envelopes)
                
                if len(tX) == 0:
                    continue
                    
                x_np, ya_np, yb_np = tX[0], tYA[0], tYB[0]
                
                chunks = chunk_trial_with_metadata(x_np, ya_np, yb_np, subject_id, trial_idx, ex.label, window_sec, window_sec, fs=64.0)
                
                for chunk in chunks:
                    x_t = torch.FloatTensor(chunk['x']).unsqueeze(0).to(device)
                    ya_t = torch.FloatTensor(chunk['ya']).unsqueeze(0).to(device)
                    yb_t = torch.FloatTensor(chunk['yb']).unsqueeze(0).to(device)
                    
                    z_eeg, z_a, z_b = model(x_t, ya_t, yb_t)
                    
                    # Compute similarity EXACTLY like train_matchnet_loso does for Pearson metrics
                    sim_a = pearson_corr(z_eeg, z_a, dim=1).mean().item()
                    sim_b = pearson_corr(z_eeg, z_b, dim=1).mean().item()
                        
                    prediction = 'A' if sim_a > sim_b else 'B'
                    
                    # wavA is always the attended stream in the preprocessed data.
                    correct = 1 if prediction == 'A' else 0
                    
                    csv_rows.append({
                        'subject_id': subject_id,
                        'trial_id': chunk['trial_id'],
                        'window_id': chunk['window_id'],
                        'sim_A': round(sim_a, 4),
                        'sim_B': round(sim_b, 4),
                        'prediction': prediction,
                        'label': 'A', # Ground truth is always A (wavA)
                        'speaker_gender': chunk['label'], # The original label (1 or 2) was just male/female
                        'correct': correct
                    })
                    
    df = pd.DataFrame(csv_rows)
    df.to_csv(out_csv, index=False)
    print(f"\nExported {len(df)} predictions to {out_csv}")
    
    acc = df['correct'].mean() * 100
    print(f"Overall CSV Accuracy Check: {acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--out_csv", type=str, default="matchnet_predictions.csv")
    parser.add_argument("--window_sec", type=float, default=10.0, help="Window duration in seconds")
    args = parser.parse_args()
    
    export_predictions(args.checkpoint_dir, args.out_csv, window_sec=args.window_sec)
