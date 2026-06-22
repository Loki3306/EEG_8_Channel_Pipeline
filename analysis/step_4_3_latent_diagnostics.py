import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.matchnet import ContrastiveMatchNet
from baselines.ridge_aad import subject_files, load_subject_examples
from training.train_matchnet_loso import prepare_dataset, get_mapping_data

def diagnostic_a_c(csv_path):
    print("--- Diagnostic A & C: Distance vs Reliability Correlations ---\n")
    df = pd.read_csv(csv_path)
    
    valid_mask = np.isfinite(df['mah_dist'])
    df = df[valid_mask].reset_index(drop=True)
    
    subj_stats = []
    for subj_id, group in df.groupby('subject_id', observed=True):
        acc = group['correct'].mean()
        mean_mah = group['mah_dist'].mean()
        try:
            auroc = roc_auc_score(group['correct'], group['margin'])
        except ValueError:
            auroc = 0.5
        subj_stats.append({
            'subject_id': subj_id,
            'acc': acc,
            'auroc': auroc,
            'mean_mah': mean_mah
        })
        
    stats_df = pd.DataFrame(subj_stats)
    
    r_acc, p_acc = pearsonr(stats_df['acc'], stats_df['mean_mah'])
    r_auc, p_auc = pearsonr(stats_df['auroc'], stats_df['mean_mah'])
    
    print(f"Diagnostic A (Accuracy vs Mean Mahalanobis): r = {r_acc:.3f} (p = {p_acc:.4f})")
    if r_acc < -0.3:
        print("  -> Moderate/Strong negative correlation! Subjects further from the training centroid are less accurate.")
    else:
        print("  -> Weak/No correlation. Mean distance doesn't strongly predict accuracy.")
        
    print(f"\nDiagnostic C (Margin AUROC vs Mean Mahalanobis): r = {r_auc:.3f} (p = {p_auc:.4f})")
    if r_auc < -0.3:
        print("  -> Moderate/Strong negative correlation! Subjects further from the centroid have worse confidence reliability.")
    else:
        print("  -> Weak/No correlation. Mean distance doesn't strongly predict confidence reliability.")
        
    print("\n--------------------------------------------------------------")

def diagnostic_b_pca(checkpoint_dir):
    print("\n--- Diagnostic B: PCA Visualization of Latent Space ---\n")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    mapping, envelopes = get_mapping_data()
    all_paths = subject_files()
    
    # We load just one model to see the latent space it learned
    subject_id = all_paths[0].stem.replace('_data_preproc', '')
    checkpoint_path = Path(checkpoint_dir) / f"matchnet_fold_{subject_id}_best.pth"
    
    if not checkpoint_path.exists():
        print(f"Cannot find checkpoint {checkpoint_path}. Skipping PCA.")
        return
        
    print(f"Loading model from {checkpoint_path.name}")
    model = ContrastiveMatchNet(eeg_model_type="eegnet", eeg_channels=8, audio_channels=28, latent_dim=64).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    z_pools = []
    subject_labels = []
    
    print("Extracting z_pool (mean-pooled embeddings) for 50 trials per subject...")
    for path in tqdm(all_paths, desc="Subjects"):
        subj = path.stem.replace('_data_preproc', '')
        exs = load_subject_examples(path)
        if not exs: continue
        
        # Limit to 50 trials to speed up extraction and balance the PCA
        exs = exs[:50]
        
        tX, tYA, tYB = prepare_dataset(exs, channels=[13, 46, 43, 23, 50, 0, 52, 14], 
                                       lowcut=1.0, highcut=6.0, subject_id=subj, 
                                       mapping=mapping, envelopes=envelopes)
                                       
        with torch.no_grad():
            for trial_idx in range(len(tX)):
                x_np = tX[trial_idx]
                
                # Split into 10s chunks roughly
                fs = 64.0
                window_samples = int(10.0 * fs)
                for start in range(0, x_np.shape[1] - window_samples, window_samples):
                    chunk = x_np[:, start:start+window_samples]
                    x_t = torch.FloatTensor(chunk).unsqueeze(0).to(device)
                    z_eeg = model.eeg_encoder(x_t) # [1, 64, 640]
                    z_pool = z_eeg.mean(dim=-1).cpu().numpy()[0] # [64]
                    
                    z_pools.append(z_pool)
                    subject_labels.append(subj)
                    
    z_pools = np.array(z_pools)
    print(f"Extracted {len(z_pools)} latent vectors. Running PCA...")
    
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(z_pools)
    
    df_pca = pd.DataFrame({
        'PC1': z_pca[:, 0],
        'PC2': z_pca[:, 1],
        'Subject': subject_labels
    })
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Subject', palette='tab20', alpha=0.6)
    plt.title(f"PCA of Latent Space (z_pool) from Fold {subject_id} Model")
    plt.tight_layout()
    out_path = Path.cwd() / "latent_pca.png"
    plt.savefig(out_path)
    print(f"PCA visualization saved to {out_path}")
    print("If subjects form distinct clusters, mean pooling did NOT destroy all subject information.")
    print("If it's a giant overlapping blob, mean pooling might be obliterating subject distinctness.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="subject_distance_predictions.csv")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    args = parser.parse_args()
    
    print("\n===========================================")
    print("STEP 4.3: LATENT SPACE DIAGNOSTICS")
    print("===========================================\n")
    print("Addressing Rank 16 issue:")
    print("The rank is exactly 16 because the EEGNet encoder bottlenecks the spatial filters to F2=16 channels")
    print("before the final 1x1 convolution projects it to 64 dimensions.")
    print("Mathematically, a 16-D space linearly projected to 64-D will always have a maximum rank of 16.")
    print("This confirms it is an architectural bottleneck, not a bug!\n")
    
    try:
        diagnostic_a_c(args.csv)
    except FileNotFoundError:
        print(f"Cannot find {args.csv}. Run export script first.")
        
    diagnostic_b_pca(args.checkpoint_dir)

if __name__ == "__main__":
    main()
