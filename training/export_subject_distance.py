import os
import sys
import argparse
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.matchnet import ContrastiveMatchNet
from baselines.ridge_aad import subject_files, load_subject_examples
from training.train_matchnet_loso import prepare_dataset, get_mapping_data

def chunk_trial_with_metadata(x, ya, yb, subject_id, trial_id, label, window_sec, hop_sec, fs=64.0):
    window_samples = int(window_sec * fs)
    hop_samples = int(hop_sec * fs)
    
    n_samples = x.shape[1]
    
    chunks = []
    for i in range(n_samples // hop_samples):
        if (i * hop_samples + window_samples) > n_samples:
            break
            
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

def get_train_distribution(model, train_paths, subject_examples, channels, lowcut, highcut, mapping, envelopes, window_sec, device):
    all_z = []
    
    for subj_path in train_paths:
        subj_id = subj_path.stem
        subj_exs = subject_examples[str(subj_path)]
        if not subj_exs: continue
        
        tX, tYA, tYB = prepare_dataset(subj_exs, channels, lowcut, highcut, subj_id, mapping, envelopes)
        
        for trial_idx in range(len(tX)):
            x_np, ya_np, yb_np = tX[trial_idx], tYA[trial_idx], tYB[trial_idx]
            ex = subj_exs[trial_idx]
            
            chunks = chunk_trial_with_metadata(x_np, ya_np, yb_np, subj_id, trial_idx, ex.label, window_sec, window_sec, fs=64.0)
            if not chunks: continue
            
            x_batch = torch.FloatTensor(np.stack([c['x'] for c in chunks])).to(device)
            with torch.no_grad():
                z_eeg = model.eeg_encoder(x_batch) # Shape: (batch, embed_dim)
                if z_eeg.dim() > 2:
                    z_eeg = z_eeg.view(z_eeg.shape[0], -1)
                all_z.append(z_eeg.cpu().numpy())
                
    if not all_z:
        return None, None, None
        
    all_z = np.concatenate(all_z, axis=0) # (Total_train_windows, embed_dim)
    embed_dim = all_z.shape[1]
    
    mu = np.mean(all_z, axis=0)
    cov = np.cov(all_z, rowvar=False)
    # Add small ridge to prevent singular matrix
    cov += np.eye(cov.shape[0]) * 1e-6
    
    return mu, cov, embed_dim

def compute_distances(z_test, mu, cov):
    diff = z_test - mu
    euc_dist = np.sqrt(np.sum(diff**2))
    
    inv_cov = np.linalg.inv(cov)
    mah_dist = np.sqrt(np.dot(np.dot(diff, inv_cov), diff))
    
    return euc_dist, mah_dist

def export_distances(checkpoint_dir, out_csv, eeg_model="eegnet", channels=[13, 46, 43, 23, 50, 0, 52, 14], lowcut=1.0, highcut=6.0, window_sec=10.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    mapping, envelopes = get_mapping_data()
    all_paths = subject_files()
    
    if not all_paths:
        print("No subjects found.")
        return
    
    print("Loading datasets...")
    subject_examples = {str(p): load_subject_examples(p) for p in all_paths}
    
    csv_rows = []
    embed_dim_recorded = None
    
    for held_out_path in tqdm(all_paths, desc="Folds"):
        subject_id = held_out_path.stem
        checkpoint_path = Path(checkpoint_dir) / f"matchnet_fold_{subject_id}_best.pth"
        
        if not checkpoint_path.exists():
            print(f"WARNING: Checkpoint {checkpoint_path} not found. Skipping subject {subject_id}.")
            continue
            
        print(f"\nProcessing Subject {subject_id} using checkpoint {checkpoint_path.name}")
        
        train_paths = [p for p in all_paths if p != held_out_path]
        
        model = ContrastiveMatchNet(eeg_model_type=eeg_model, eeg_channels=len(channels), audio_channels=28, latent_dim=64).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        
        # 1. Compute training distribution
        mu, cov, embed_dim = get_train_distribution(
            model, train_paths, subject_examples, channels, lowcut, highcut, 
            mapping, envelopes, window_sec, device
        )
        
        if embed_dim_recorded is None and embed_dim is not None:
            embed_dim_recorded = embed_dim
            print(f"Embedding Dimensionality: {embed_dim_recorded}")
            
        if mu is None:
            continue
            
        # 2. Evaluate held-out subject
        test_exs = subject_examples[str(held_out_path)]
        tX, tYA, tYB = prepare_dataset(test_exs, channels, lowcut, highcut, subject_id, mapping, envelopes)
        
        with torch.no_grad():
            for trial_idx in range(len(tX)):
                x_np, ya_np, yb_np = tX[trial_idx], tYA[trial_idx], tYB[trial_idx]
                ex = test_exs[trial_idx]
                
                chunks = chunk_trial_with_metadata(x_np, ya_np, yb_np, subject_id, trial_idx, ex.label, window_sec, window_sec, fs=64.0)
                
                for chunk in chunks:
                    x_t = torch.FloatTensor(chunk['x']).unsqueeze(0).to(device)
                    ya_t = torch.FloatTensor(chunk['ya']).unsqueeze(0).to(device)
                    yb_t = torch.FloatTensor(chunk['yb']).unsqueeze(0).to(device)
                    
                    z_eeg, z_a, z_b = model(x_t, ya_t, yb_t)
                    
                    if z_eeg.dim() > 2:
                        z_eeg_flat = z_eeg.view(z_eeg.shape[0], -1)
                    else:
                        z_eeg_flat = z_eeg
                        
                    z_eeg_np = z_eeg_flat.cpu().numpy()[0]
                    
                    euc_dist, mah_dist = compute_distances(z_eeg_np, mu, cov)
                    
                    sim_a = pearson_corr(z_eeg, z_a, dim=1).mean().item()
                    sim_b = pearson_corr(z_eeg, z_b, dim=1).mean().item()
                        
                    prediction = 'A' if sim_a > sim_b else 'B'
                    correct = 1 if prediction == 'A' else 0
                    
                    margin = abs(sim_a - sim_b)
                    
                    csv_rows.append({
                        'subject_id': subject_id,
                        'trial_id': chunk['trial_id'],
                        'window_id': chunk['window_id'],
                        'sim_A': round(sim_a, 4),
                        'sim_B': round(sim_b, 4),
                        'margin': round(margin, 4),
                        'prediction': prediction,
                        'correct': correct,
                        'euc_dist': round(euc_dist, 4),
                        'mah_dist': round(mah_dist, 4)
                    })
                    
    df = pd.DataFrame(csv_rows)
    df.to_csv(out_csv, index=False)
    print(f"\nExported {len(df)} predictions with subject distance to {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--out_csv', type=str, default='subject_distance_predictions.csv')
    args = parser.parse_args()
    
    export_distances(args.checkpoint_dir, args.out_csv)
