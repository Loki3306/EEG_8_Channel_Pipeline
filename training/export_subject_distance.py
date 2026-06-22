import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.matchnet import MatchNet
from utils.data_loading import load_bids_dataset_lazy, prepare_dataset
from utils.chunking import chunk_trial_with_metadata
from utils.loss import pearson_corr

def get_train_distribution(model, train_subjects, subject_examples, channels, lowcut, highcut, mapping, envelopes, window_sec, device):
    all_z = []
    
    # We process each train subject to collect their embeddings
    for subj_id in train_subjects:
        subj_exs = subject_examples[str(subj_id)]
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
        return None, None
        
    all_z = np.concatenate(all_z, axis=0) # (Total_train_windows, embed_dim)
    
    # Print embed dim
    embed_dim = all_z.shape[1]
    
    mu = np.mean(all_z, axis=0)
    cov = np.cov(all_z, rowvar=False)
    # Add small ridge to prevent singular matrix
    cov += np.eye(cov.shape[0]) * 1e-6
    
    return mu, cov, embed_dim

def compute_distances(z_test, mu, cov):
    # z_test is shape (embed_dim,)
    diff = z_test - mu
    
    # Euclidean
    euc_dist = np.sqrt(np.sum(diff**2))
    
    # Mahalanobis
    inv_cov = np.linalg.inv(cov)
    mah_dist = np.sqrt(np.dot(np.dot(diff, inv_cov), diff))
    
    return euc_dist, mah_dist

def export_distances(checkpoint_dir, out_csv):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    bids_root = "ds007136"
    print("Loading dataset metadata...")
    subject_examples, channels, mapping, envelopes = load_bids_dataset_lazy(bids_root)
    
    lowcut, highcut = 1.0, 9.0
    window_sec = 10.0
    
    csv_rows = []
    
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = list(checkpoint_dir.glob("matchnet_loso_*.pt"))
    print(f"Found {len(checkpoints)} checkpoints.")
    
    embed_dim_recorded = None
    
    for ckpt_path in tqdm(checkpoints, desc="Folds"):
        held_out_path = ckpt_path.stem.replace("matchnet_loso_", "")
        subject_id = held_out_path.split('_')[0]
        
        # Determine train subjects
        all_subjs = list(subject_examples.keys())
        train_subjs = [s for s in all_subjs if s != held_out_path]
        
        model = MatchNet(in_channels=len(channels)).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.eval()
        
        # 1. Compute training distribution
        mu, cov, embed_dim = get_train_distribution(
            model, train_subjs, subject_examples, channels, lowcut, highcut, 
            mapping, envelopes, window_sec, device
        )
        
        if embed_dim_recorded is None:
            embed_dim_recorded = embed_dim
            print(f"\nEmbedding Dimensionality: {embed_dim_recorded}")
            
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
                    
                    z_eeg = model.eeg_encoder(x_t)
                    z_a = model.audio_encoder(ya_t)
                    z_b = model.audio_encoder(yb_t)
                    
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
