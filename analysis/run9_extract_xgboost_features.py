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

def corrupt_eeg(eeg, wav_a, wav_b, mode, device):
    if mode == "clean": return eeg, wav_a, wav_b
    elif mode == "random": return torch.randn_like(eeg), wav_a, wav_b
    elif mode == "zero": return torch.zeros_like(eeg), wav_a, wav_b
    elif mode == "gaussian": return eeg + torch.randn_like(eeg) * 0.5, wav_a, wav_b
    elif mode == "audio_permute": return eeg, wav_b, wav_a
    elif mode == "label_shuffle": return eeg, wav_a[:, :, torch.randperm(wav_a.shape[-1])], wav_b[:, :, torch.randperm(wav_b.shape[-1])]
    elif mode == "circular_shift": 
        shift = eeg.shape[-1] // 2
        return torch.roll(eeg, shifts=shift, dims=-1), wav_a, wav_b
    return eeg, wav_a, wav_b

def get_predictions_for_xgb(model, eeg, wav_a, wav_b, win_samples, hop_samples):
    features_list = []
    for start in range(0, eeg.shape[-1] - win_samples + 1, hop_samples):
        stop = start + win_samples
        eeg_win = eeg[:, :, start:stop]
        
        pred, z_pool = model(eeg_win, return_features=True)
        pred_np = pred.squeeze(0).cpu().numpy()
        
        wa = wav_a[:, :, start:stop].squeeze(1).squeeze(0).cpu().numpy()
        wb = wav_b[:, :, start:stop].squeeze(1).squeeze(0).cpu().numpy()
        
        ca = safe_corr_np(pred_np, wa)
        cb = safe_corr_np(pred_np, wb)
        margin = ca - cb
        
        z_np = z_pool.squeeze(0).cpu().numpy()
        z_norm = np.linalg.norm(z_np)
        z_std = np.std(z_np)
        
        correct = 1 if margin > 0 else 0
        feat = {
            'ca': ca, 'cb': cb, 'margin': margin, 
            'latent_norm': z_norm, 'latent_std': z_std, 'correct': correct
        }
        for i, val in enumerate(z_np):
            feat[f'z_{i}'] = val
        features_list.append(feat)
    return features_list

import argparse

def main():
    parser = argparse.ArgumentParser(description="Phase 9: Extract XGBoost Features")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    args = parser.parse_args()

    print("--- Phase 9: Feature Extraction for XGBoost ---")
    if args.cache_dir: cache_dir = Path(args.cache_dir)
    else: cache_dir = Path("/kaggle/input/datasets/lokeshgile/kul-processed/data/processed_kul") if Path("/kaggle/input/datasets/lokeshgile/kul-processed/data/processed_kul").exists() else REPO_ROOT / "data" / "processed_kul"
            
    if args.checkpoint_dir: checkpoint_dir = Path(args.checkpoint_dir)
    else: checkpoint_dir = REPO_ROOT / "results" / "run7_multitask_conformer_loso" / "checkpoints" / "seed_1"
        
    out_dir = REPO_ROOT / "results" / "run9_xgboost" / "features"
    os.makedirs(out_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using cache_dir: {cache_dir}\nUsing checkpoint_dir: {checkpoint_dir}\nUsing device: {device}")
    
    loader = KULCachedLoader(cache_dir)
    loader.load_all()
    subjects = list(loader.subjects_data.keys())
    win_samples, hop_samples = 640, 64
    
    modes = ["clean", "random", "zero", "gaussian", "audio_permute", "label_shuffle", "circular_shift"]
    
    for test_subj in subjects:
        ckpt_path = checkpoint_dir / f"model_{test_subj}.pt"
        if not ckpt_path.exists(): continue
            
        print(f"\nProcessing Fold (Test Subject = {test_subj})...")
        model = AADConformer(in_channels=8).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        
        train_subjects = [s for s in subjects if s != test_subj]
        
        # 1. Extract Training Features (Clean Only)
        train_features = []
        with torch.no_grad():
            for t_subj in train_subjects:
                for t in loader.subjects_data[t_subj]:
                    eeg = normalize_eeg(t["eeg"].unsqueeze(0).to(device))
                    wav_a = normalize_audio(t["audio_a"].unsqueeze(0).to(device).mean(dim=1, keepdim=True))
                    wav_b = normalize_audio(t["audio_b"].unsqueeze(0).to(device).mean(dim=1, keepdim=True))
                    min_len = min(eeg.shape[-1], wav_a.shape[-1], wav_b.shape[-1])
                    
                    feats = get_predictions_for_xgb(model, eeg[:,:,:min_len], wav_a[:,:,:min_len], wav_b[:,:,:min_len], win_samples, hop_samples)
                    for f in feats: f['source_subject'] = t_subj
                    train_features.extend(feats)
        pd.DataFrame(train_features).to_csv(out_dir / f"fold_{test_subj}_train_clean.csv", index=False)
        print(f"  -> Saved {len(train_features)} train windows.")
        
        # 2. Extract Test Features (All Modes)
        for mode in modes:
            test_features = []
            with torch.no_grad():
                for t in loader.subjects_data[test_subj]:
                    eeg_base = t["eeg"].unsqueeze(0).to(device)
                    wav_a_base = t["audio_a"].unsqueeze(0).to(device).mean(dim=1, keepdim=True)
                    wav_b_base = t["audio_b"].unsqueeze(0).to(device).mean(dim=1, keepdim=True)
                    
                    eeg, wav_a, wav_b = corrupt_eeg(eeg_base, wav_a_base, wav_b_base, mode, device)
                    eeg, wav_a, wav_b = normalize_eeg(eeg), normalize_audio(wav_a), normalize_audio(wav_b)
                    min_len = min(eeg.shape[-1], wav_a.shape[-1], wav_b.shape[-1])
                    
                    feats = get_predictions_for_xgb(model, eeg[:,:,:min_len], wav_a[:,:,:min_len], wav_b[:,:,:min_len], win_samples, hop_samples)
                    for f in feats: f['source_subject'] = test_subj
                    test_features.extend(feats)
            pd.DataFrame(test_features).to_csv(out_dir / f"fold_{test_subj}_test_{mode}.csv", index=False)
            print(f"  -> Saved {len(test_features)} test windows ({mode}).")

if __name__ == "__main__":
    main()
