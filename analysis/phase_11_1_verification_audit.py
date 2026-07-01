import os
import sys
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.aad_conformer import AADConformer
from data.kul_cached_dataset import KULCachedLoader
from analysis.interpretability.utils import safe_corr_np, normalize_eeg, normalize_audio

def run_verification_audit(subject="S1", ckpt_path=None):
    print("=" * 80)
    print("PHASE 11.1 — VERIFICATION AUDIT")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if not ckpt_path:
        ckpt_path = REPO_ROOT / "results" / "run7_multitask_conformer_loso" / "checkpoints" / "seed_1" / f"model_{subject}.pt"
        if not ckpt_path.exists():
            ckpt_path = Path(f"/kaggle/working/EEG_8_Channel_Pipeline/results/run7_multitask_conformer_loso/checkpoints/seed_1/model_{subject}.pt")
            
    print(f"\n1. MODEL CHECKPOINT VERIFICATION")
    print("-" * 40)
    model = AADConformer(in_channels=8).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    
    # Check if confidence_head weights exist and are not zero
    has_conf_head = any("confidence_head" in k for k in state_dict.keys())
    print(f"Confidence Head found in state_dict: {has_conf_head}")
    
    if has_conf_head:
        # Load state dict
        model.load_state_dict(state_dict)
        weight_norm = torch.norm(model.confidence_head[0].weight).item()
        print(f"Confidence Head Linear(1) Weight Norm: {weight_norm:.4f}")
    else:
        print("WARNING: Confidence Head missing from checkpoint!")
        
    print(f"\n2. EVALUATION MODE VERIFICATION")
    print("-" * 40)
    model.eval()
    print(f"Model training mode: {model.training}")
    print(f"Dropout modules in eval mode: {all(not m.training for m in model.modules() if isinstance(m, torch.nn.Dropout))}")
    
    print(f"\n3. DATA NORMALIZATION VERIFICATION & 5. BASELINE ACCURACY")
    print("-" * 40)
    
    cache_dir = Path("/kaggle/input/datasets/lokeshgile/kul-processed/data/processed_kul")
    if not cache_dir.exists():
        cache_dir = REPO_ROOT / "data" / "processed_kul"
        
    loader = KULCachedLoader(cache_dir)
    all_data = loader.load_all()
    data = all_data[subject]
    
    print(f"Loaded {len(data)} trials for {subject}.")
    
    win_samples = 320 # 5s at 64Hz
    hop_samples = 64  # 1s hop
    
    all_margins_unnorm = []
    all_margins_norm = []
    
    correct_trials_norm = 0
    correct_trials_unnorm = 0
    
    confidences_norm = []
    confidences_unnorm = []
    
    print("Running inference over all trials...")
    with torch.no_grad():
        for t_idx, t in enumerate(data):
            eeg = t["eeg"].unsqueeze(0).to(device) # (1, 8, Time)
            wav_a = t["audio_a"].unsqueeze(0).to(device).mean(dim=1, keepdim=True) # (1, 1, Time)
            wav_b = t["audio_b"].unsqueeze(0).to(device).mean(dim=1, keepdim=True) # (1, 1, Time)
            
            # Unnormalized (Phase 11 buggy approach)
            eeg_unnorm = eeg.clone()
            
            # Normalized (Phase 7 correct approach)
            eeg_norm = normalize_eeg(eeg)
            wav_a_norm = normalize_audio(wav_a)
            wav_b_norm = normalize_audio(wav_b)
            
            win_correct_norm = 0
            win_correct_unnorm = 0
            num_windows = 0
            
            for start in range(0, eeg.shape[-1] - win_samples + 1, hop_samples):
                stop = start + win_samples
                num_windows += 1
                
                # --- NORM (Correct) ---
                e_win_norm = eeg_norm[:, :, start:stop]
                pred_norm, z_norm = model(e_win_norm, return_features=True)
                
                wa_win = wav_a_norm[:, :, start:stop].squeeze().cpu().numpy()
                wb_win = wav_b_norm[:, :, start:stop].squeeze().cpu().numpy()
                pn = pred_norm.squeeze().cpu().numpy()
                
                ca_n = safe_corr_np(pn, wa_win)
                cb_n = safe_corr_np(pn, wb_win)
                
                # Phase 7 margin formula: corr_a - corr_b
                margin_n = ca_n - cb_n 
                
                conf_n = model.predict_confidence(
                    z_norm, 
                    torch.tensor([ca_n], dtype=torch.float32, device=device),
                    torch.tensor([cb_n], dtype=torch.float32, device=device),
                    torch.tensor([margin_n], dtype=torch.float32, device=device)
                ).item()
                
                if margin_n > 0: win_correct_norm += 1
                
                all_margins_norm.append(margin_n)
                confidences_norm.append(conf_n)
                
                # --- UNNORM (Phase 11 Buggy) ---
                e_win_unnorm = eeg_unnorm[:, :, start:stop]
                pred_unnorm, z_unnorm = model(e_win_unnorm, return_features=True)
                
                wa_unnorm = wav_a[:, :, start:stop].squeeze().cpu().numpy()
                wb_unnorm = wav_b[:, :, start:stop].squeeze().cpu().numpy()
                pu = pred_unnorm.squeeze().cpu().numpy()
                
                ca_u = safe_corr_np(pu, wa_unnorm)
                cb_u = safe_corr_np(pu, wb_unnorm)
                
                # Phase 11 buggy margin formula: sim_b - sim_a
                margin_u = cb_u - ca_u 
                
                conf_u = model.predict_confidence(
                    z_unnorm, 
                    torch.tensor([ca_u], dtype=torch.float32, device=device),
                    torch.tensor([cb_u], dtype=torch.float32, device=device),
                    torch.tensor([margin_u], dtype=torch.float32, device=device)
                ).item()
                
                if ca_u > cb_u: win_correct_unnorm += 1
                
                all_margins_unnorm.append(margin_u)
                confidences_unnorm.append(conf_u)
                
            if win_correct_norm > num_windows / 2:
                correct_trials_norm += 1
            if win_correct_unnorm > num_windows / 2:
                correct_trials_unnorm += 1
                
    print(f"\n4 & 5. MARGIN AND BASELINE ACCURACY RESULTS")
    print("-" * 40)
    print("UNNORMALIZED + BUGGY MARGIN (Phase 11 Pipeline):")
    print(f"  Trial Accuracy: {correct_trials_unnorm}/{len(data)} ({correct_trials_unnorm/len(data)*100:.1f}%)")
    print(f"  Mean Margin (Inverted): {np.mean(all_margins_unnorm):.4f}")
    
    print("\nNORMALIZED + CORRECT MARGIN (Phase 7 Pipeline):")
    print(f"  Trial Accuracy: {correct_trials_norm}/{len(data)} ({correct_trials_norm/len(data)*100:.1f}%)")
    print(f"  Mean Margin: {np.mean(all_margins_norm):.4f}")
    
    print(f"\n6. CONFIDENCE DISTRIBUTION SNAPSHOT")
    print("-" * 40)
    print("UNNORMALIZED (Buggy):")
    print(f"  Mean: {np.mean(confidences_unnorm):.4f}")
    print(f"  Std:  {np.std(confidences_unnorm):.4f}")
    print(f"  Min:  {np.min(confidences_unnorm):.4f}")
    print(f"  Max:  {np.max(confidences_unnorm):.4f}")
    
    print("\nNORMALIZED (Correct):")
    print(f"  Mean: {np.mean(confidences_norm):.4f}")
    print(f"  Std:  {np.std(confidences_norm):.4f}")
    print(f"  Min:  {np.min(confidences_norm):.4f}")
    print(f"  Max:  {np.max(confidences_norm):.4f}")

if __name__ == "__main__":
    run_verification_audit()
