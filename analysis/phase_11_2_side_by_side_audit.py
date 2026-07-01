import os
import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.aad_conformer import AADConformer
from data.kul_cached_dataset import KULCachedLoader
from analysis.interpretability.utils import safe_corr_np, normalize_eeg, normalize_audio

def print_stats(name, tensor):
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if not isinstance(tensor, torch.Tensor):
        print(f"{name:.<30} {type(tensor)}")
        return
        
    t = tensor.float()
    print(f"{name:.<30} Shape: {list(t.shape):<20} Mean: {t.mean().item():>8.4f}  Std: {t.std().item():>8.4f}  Min: {t.min().item():>8.4f}  Max: {t.max().item():>8.4f}")

def run_side_by_side_audit(subject="S1", ckpt_path=None):
    print("=" * 100)
    print("PHASE 11.2 — SIDE-BY-SIDE TENSOR AUDIT")
    print("=" * 100)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if not ckpt_path:
        ckpt_path = REPO_ROOT / "results" / "run7_multitask_conformer_loso" / "checkpoints" / "seed_1" / f"model_{subject}.pt"
        if not ckpt_path.exists():
            ckpt_path = Path(f"/kaggle/working/EEG_8_Channel_Pipeline/results/run7_multitask_conformer_loso/checkpoints/seed_1/model_{subject}.pt")
            
    model = AADConformer(in_channels=8).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    cache_dir = Path("/kaggle/input/datasets/lokeshgile/kul-processed/data/processed_kul")
    if not cache_dir.exists():
        cache_dir = REPO_ROOT / "data" / "processed_kul"
        
    loader = KULCachedLoader(cache_dir)
    data = loader.load_all()[subject]
    
    # We will trace Trial 0
    t = data[0]
    
    # Raw trial tensors
    eeg_raw = t["eeg"].unsqueeze(0).to(device) # (1, 8, Time)
    wav_a_raw = t["audio_a"].unsqueeze(0).to(device).mean(dim=1, keepdim=True) # (1, 1, Time)
    wav_b_raw = t["audio_b"].unsqueeze(0).to(device).mean(dim=1, keepdim=True) # (1, 1, Time)
    
    # Hooks to intercept activations
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
        
    # Register hooks
    model.temporal_conv.register_forward_hook(get_activation('temporal_conv'))
    model.spatial_conv.register_forward_hook(get_activation('spatial_conv'))
    model.conformer_blocks[-1].register_forward_hook(get_activation('conformer_block'))
    
    # =========================================================================
    # PIPELINE 1: PHASE 7 EVALUATION LOGIC
    # =========================================================================
    print("\n" + "=" * 80)
    print("PIPELINE 1: PHASE 7 EVALUATION LOGIC (10s window, full-trial normalization)")
    print("=" * 80)
    
    fs = 64
    win_samples_p1 = 10 * fs
    
    # Phase 7 eval logic normalizes the full trial FIRST
    eeg_norm = normalize_eeg(eeg_raw)
    wav_a_norm = normalize_audio(wav_a_raw)
    wav_b_norm = normalize_audio(wav_b_raw)
    
    # Extract first window
    eeg_win_p1 = eeg_norm[:, :, :win_samples_p1]
    wa_win_p1 = wav_a_norm[:, :, :win_samples_p1]
    wb_win_p1 = wav_b_norm[:, :, :win_samples_p1]
    
    print_stats("Input EEG (Normalized)", eeg_win_p1)
    print_stats("Input Audio A (Normalized)", wa_win_p1)
    
    activations.clear()
    with torch.no_grad():
        pred_p1, z_pool_p1 = model(eeg_win_p1, return_features=True)
        
        print_stats("Temporal Conv Output", activations['temporal_conv'])
        print_stats("Spatial Conv Output", activations['spatial_conv'])
        print_stats("Transformer Output", activations['conformer_block'])
        print_stats("Global Pooling (z_pool)", z_pool_p1)
        print_stats("Regression Head Output", pred_p1)
        
        wa_np_p1 = wa_win_p1.squeeze().cpu().numpy()
        wb_np_p1 = wb_win_p1.squeeze().cpu().numpy()
        pred_np_p1 = pred_p1.squeeze().cpu().numpy()
        
        ca_p1 = safe_corr_np(pred_np_p1, wa_np_p1)
        cb_p1 = safe_corr_np(pred_np_p1, wb_np_p1)
        margin_p1 = ca_p1 - cb_p1 # Phase 7 margin formula
        
        print(f"\nPearson A (ca): {ca_p1:.4f}")
        print(f"Pearson B (cb): {cb_p1:.4f}")
        print(f"Margin (ca - cb): {margin_p1:.4f}")
        
        conf_p1 = model.predict_confidence(
            z_pool_p1, 
            torch.tensor([ca_p1], dtype=torch.float32, device=device),
            torch.tensor([cb_p1], dtype=torch.float32, device=device),
            torch.tensor([margin_p1], dtype=torch.float32, device=device)
        ).item()
        
        print(f"\nConfidence Head Probability: {conf_p1:.4f}")

    # =========================================================================
    # PIPELINE 2: PHASE 11 BUGGY LOGIC
    # =========================================================================
    print("\n" + "=" * 80)
    print("PIPELINE 2: PHASE 11 BUGGY LOGIC (5s window, no normalization, inverted margin)")
    print("=" * 80)
    
    win_samples_p2 = 320 # 5s
    
    # Phase 11 buggy logic extracts raw window without normalizing
    eeg_win_p2 = eeg_raw[:, :, :win_samples_p2]
    wa_win_p2 = wav_a_raw[:, :, :win_samples_p2]
    wb_win_p2 = wav_b_raw[:, :, :win_samples_p2]
    
    print_stats("Input EEG (Raw)", eeg_win_p2)
    print_stats("Input Audio A (Raw)", wa_win_p2)
    
    activations.clear()
    with torch.no_grad():
        pred_p2, z_pool_p2 = model(eeg_win_p2, return_features=True)
        
        print_stats("Temporal Conv Output", activations['temporal_conv'])
        print_stats("Spatial Conv Output", activations['spatial_conv'])
        print_stats("Transformer Output", activations['transformer'])
        print_stats("Global Pooling (z_pool)", z_pool_p2)
        print_stats("Regression Head Output", pred_p2)
        
        # In phase 11, the audio target used for correlation WAS 
        # normalized because the model outputs pred_c, ya_c, yb_c
        # Wait, the code subtracted the mean, but did not divide by std:
        # pred_c = pred - pred.mean()
        # ya_c = wa - wa.mean()
        # So we just compute pearson. safe_corr_np is scale invariant anyway.
        wa_np_p2 = wa_win_p2.squeeze().cpu().numpy()
        wb_np_p2 = wb_win_p2.squeeze().cpu().numpy()
        pred_np_p2 = pred_p2.squeeze().cpu().numpy()
        
        ca_p2 = safe_corr_np(pred_np_p2, wa_np_p2)
        cb_p2 = safe_corr_np(pred_np_p2, wb_np_p2)
        margin_p2 = cb_p2 - ca_p2 # Phase 11 inverted margin formula
        
        print(f"\nPearson A (ca): {ca_p2:.4f}")
        print(f"Pearson B (cb): {cb_p2:.4f}")
        print(f"Margin (cb - ca): {margin_p2:.4f}")
        
        conf_p2 = model.predict_confidence(
            z_pool_p2, 
            torch.tensor([ca_p2], dtype=torch.float32, device=device),
            torch.tensor([cb_p2], dtype=torch.float32, device=device),
            torch.tensor([margin_p2], dtype=torch.float32, device=device)
        ).item()
        
        print(f"\nConfidence Head Probability: {conf_p2:.4f}")


if __name__ == "__main__":
    run_side_by_side_audit()
