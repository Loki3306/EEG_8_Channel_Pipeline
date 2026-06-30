import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.kul_cached_dataset import KULCachedLoader
from models.aad_conformer import AADConformer
from analysis.interpretability.utils import safe_corr_np, normalize_eeg, normalize_audio

warnings.filterwarnings("ignore")

def sliding_window_extraction(model, eeg, wav_a, wav_b, win_samples, hop_samples):
    """Extract latent features, correlations, margins, and targets."""
    features = []
    
    for start in range(0, eeg.shape[-1] - win_samples + 1, hop_samples):
        stop = start + win_samples
        eeg_win = eeg[:, :, start:stop]
        wa = wav_a[:, :, start:stop].squeeze(1).squeeze(0).cpu().numpy()
        wb = wav_b[:, :, start:stop].squeeze(1).squeeze(0).cpu().numpy()
        
        # Forward pass returning features
        # AADConformer forward signature: (x, return_confidence, return_features)
        pred, z_pool = model(eeg_win, return_confidence=False, return_features=True)
        pred = pred.squeeze(0).cpu().numpy()
        
        # We know z_pool is [1, embed_dim]
        z_pool_np = z_pool.squeeze(0).cpu().numpy()
        
        ca = safe_corr_np(pred, wa)
        cb = safe_corr_np(pred, wb)
        
        margin = ca - cb
        correct = int(margin > 0)
        
        feat_dict = {
            'corr_a': ca,
            'corr_b': cb,
            'margin': margin,
            'correct': correct,
            'embedding_norm': np.linalg.norm(z_pool_np),
        }
        
        # Add all 64 latent dimensions as separate columns for XGBoost/Feature analysis
        for i, val in enumerate(z_pool_np):
            feat_dict[f'z_{i}'] = val
            
        features.append(feat_dict)
        
    return features

def extract_subject_features(model, test_trials, device, win_samples, hop_samples):
    all_features = []
    
    model.eval()
    with torch.no_grad():
        for t in test_trials:
            eeg = t["eeg"].unsqueeze(0).to(device)
            wav_a = t["audio_a"].unsqueeze(0).to(device).mean(dim=1, keepdim=True)
            wav_b = t["audio_b"].unsqueeze(0).to(device).mean(dim=1, keepdim=True)
            
            eeg = normalize_eeg(eeg)
            wav_a = normalize_audio(wav_a)
            wav_b = normalize_audio(wav_b)
            
            min_len = min(eeg.shape[-1], wav_a.shape[-1], wav_b.shape[-1])
            eeg, wav_a, wav_b = eeg[:,:,:min_len], wav_a[:,:,:min_len], wav_b[:,:,:min_len]
            
            feats = sliding_window_extraction(model, eeg, wav_a, wav_b, win_samples, hop_samples)
            all_features.extend(feats)
            
    return all_features

def main():
    print("--- Phase 7: Generating Learned Confidence Dataset ---")
    
    cache_dir = REPO_ROOT / "data" / "processed_kul"
    if Path("/kaggle/input/datasets/lowk1ee/kul-preprocessed-cache/data/processed_kul").exists():
        cache_dir = Path("/kaggle/input/datasets/lowk1ee/kul-preprocessed-cache/data/processed_kul")
    
    checkpoint_dir = REPO_ROOT / "conformer_loso_results" / "checkpoints" / "seed_1"
    kaggle_ckpt = Path("/kaggle/input/datasets/lowkieee/eeg-aad-conformer-seed1-checkpoints/checkpoints/seed_1")
    if not kaggle_ckpt.exists():
        kaggle_ckpt = Path("/kaggle/input/eeg-aad-conformer-seed1-checkpoints/checkpoints/seed_1")
    if kaggle_ckpt.exists():
        checkpoint_dir = kaggle_ckpt
        
    out_dir = REPO_ROOT / "results" / "run7_learned_confidence"
    os.makedirs(out_dir, exist_ok=True)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    loader = KULCachedLoader(cache_dir)
    loader.load_all()
    
    subjects_to_test = list(loader.subjects_data.keys())
    print(f"Evaluating {len(subjects_to_test)} subjects...")
    
    fs = 64
    win_samples = 10 * fs
    hop_samples = fs
    
    all_data = []
    
    for subj in subjects_to_test:
        print(f"Processing Subject {subj}...")
        ckpt_path = checkpoint_dir / f"model_{subj}.pt"
        if not ckpt_path.exists():
            print(f"  [Error] Checkpoint not found: {ckpt_path}. Skipping.")
            continue
            
        model = AADConformer(in_channels=8).to(device)
        # We load strictly the regression weights (ignoring the randomly initialized new confidence head)
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        test_trials = loader.subjects_data[subj]
        feats = extract_subject_features(model, test_trials, device, win_samples, hop_samples)
        
        for f in feats:
            f['subject'] = subj
            all_data.append(f)
            
    df = pd.DataFrame(all_data)
    
    csv_path = out_dir / "confidence_features.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"\nDataset saved to {csv_path}")
    print(f"Total samples: {len(df)}")
    print(f"Class balance (Correct predictions): {df['correct'].mean() * 100:.2f}%")

if __name__ == "__main__":
    main()
