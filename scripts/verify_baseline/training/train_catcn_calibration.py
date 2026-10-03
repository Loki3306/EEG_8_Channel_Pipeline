import argparse
import sys
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from copy import deepcopy
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.catcn import CATCNDirectDecoder
from baselines.ridge_aad import load_subject_examples, subject_files
from training.train_matchnet_wavlm import (
    FS, TRAIN_WINDOW_SEC,
    prepare_dataset, get_mapping_data
)
from training.montages import MONTAGES, DTU_CHANNELS

@torch.no_grad()
def evaluate_catcn_multiwindow_batched(model, X, Y_A, Y_B, device, batch_size=256):
    """
    Batched multi-window evaluation across trials.
    Returns dictionary with accumulated 1s accuracy at 5s, 10s, 20s, and 40s windows.
    """
    model.eval()
    samples_1s = int(1 * FS)
    
    trial_sub_1s = []
    
    for i in range(len(X)):
        x_np, ya_np, yb_np = X[i], Y_A[i], Y_B[i]
        trial_len = x_np.shape[1]
        
        c_1s_x, c_1s_ya, c_1s_yb = [], [], []
        start = 0
        while start + samples_1s <= trial_len:
            end = start + samples_1s
            c_1s_x.append(x_np[:, start:end])
            c_1s_ya.append(ya_np[:, start:end])
            c_1s_yb.append(yb_np[:, start:end])
            start += samples_1s
            
        if c_1s_x:
            bx = torch.from_numpy(np.stack(c_1s_x, axis=0)).to(device, dtype=torch.float32)
            bya = torch.from_numpy(np.stack(c_1s_ya, axis=0)).to(device, dtype=torch.float32)
            byb = torch.from_numpy(np.stack(c_1s_yb, axis=0)).to(device, dtype=torch.float32)
            
            delta_list = []
            for b_idx in range(0, bx.size(0), batch_size):
                d_b, _, _ = model(bx[b_idx:b_idx+batch_size], bya[b_idx:b_idx+batch_size], byb[b_idx:b_idx+batch_size])
                delta_list.append(d_b.detach().cpu())
            deltas = torch.cat(delta_list).numpy()
            d_1s = deltas.tolist()
            v_1s = [1.0 if d > 0 else (0.5 if d == 0 else 0.0) for d in d_1s]
        else:
            d_1s, v_1s = [], []
        trial_sub_1s.append((d_1s, v_1s))

    results = {}
    for w in [5, 10, 20, 40]:
        c_accum_1s, n_accum_1s = 0.0, 0
        m_1s = w
        for d_list, v_list in trial_sub_1s:
            b_start = 0
            while b_start + m_1s <= len(d_list):
                block_d = d_list[b_start : b_start + m_1s]
                if sum(block_d) > 0: c_accum_1s += 1.0
                elif sum(block_d) == 0: c_accum_1s += 0.5
                n_accum_1s += 1
                b_start += m_1s
        acc = c_accum_1s / max(n_accum_1s, 1) if n_accum_1s > 0 else 0.5
        results[w] = acc
        
    return results

def train_generic_backbone(train_paths, montage_channels, args, device, mapping, envelopes):
    """
    Pre-trains a generic cross-subject CA-TCN backbone on the 17 training subjects.
    """
    win_samples = int(TRAIN_WINDOW_SEC * FS)
    hop_samples = int(args.train_hop_sec * FS)
    
    tr_x_list, tr_ya_list, tr_yb_list = [], [], []
    va_x_list, va_ya_list, va_yb_list = [], [], []
    
    print(f"  -> Loading & pre-chunking training data from {len(train_paths)} subjects...")
    for p in train_paths:
        sub_name = p.stem
        exs = list(load_subject_examples(p))
        X_sub, YA_sub, YB_sub = prepare_dataset(
            exs, montage_channels, args.lowcut, args.highcut, sub_name, mapping, envelopes
        )
        YA_sub = [ya.mean(axis=0, keepdims=True).astype(np.float32) if ya.shape[0] > 1 else ya.astype(np.float32) for ya in YA_sub]
        YB_sub = [yb.mean(axis=0, keepdims=True).astype(np.float32) if yb.shape[0] > 1 else yb.astype(np.float32) for yb in YB_sub]
        
        n_trials = len(X_sub)
        val_split = max(1, int(0.1 * n_trials))
        rng = np.random.RandomState(42)
        perm = rng.permutation(n_trials)
        
        for idx in perm[val_split:]:
            x, ya, yb = X_sub[idx], YA_sub[idx], YB_sub[idx]
            t_len = x.shape[1]
            start = 0
            while start + win_samples <= t_len:
                end = start + win_samples
                tr_x_list.append(x[:, start:end])
                tr_ya_list.append(ya[:, start:end])
                tr_yb_list.append(yb[:, start:end])
                start += hop_samples
                
        for idx in perm[:val_split]:
            x, ya, yb = X_sub[idx], YA_sub[idx], YB_sub[idx]
            t_len = x.shape[1]
            start = 0
            while start + win_samples <= t_len:
                end = start + win_samples
                va_x_list.append(x[:, start:end])
                va_ya_list.append(ya[:, start:end])
                va_yb_list.append(yb[:, start:end])
                start += hop_samples

    X_tr_t = torch.from_numpy(np.stack(tr_x_list, axis=0))
    YA_tr_t = torch.from_numpy(np.stack(tr_ya_list, axis=0))
    YB_tr_t = torch.from_numpy(np.stack(tr_yb_list, axis=0))
    
    X_va_t = torch.from_numpy(np.stack(va_x_list, axis=0))
    YA_va_t = torch.from_numpy(np.stack(va_ya_list, axis=0))
    YB_va_t = torch.from_numpy(np.stack(va_yb_list, axis=0))
    
    train_ds = TensorDataset(X_tr_t, YA_tr_t, YB_tr_t)
    val_ds = TensorDataset(X_va_t, YA_va_t, YB_va_t)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=0)
    
    model = CATCNDirectDecoder(
        eeg_channels=len(montage_channels),
        audio_channels=1,
        hidden_dim=64,
        max_lag_samples=8,
        dropout=0.2
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr_pretrain, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_pretrain, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    best_val_loss = float('inf')
    best_weights = deepcopy(model.state_dict())
    
    t0 = datetime.now()
    for epoch in range(args.epochs_pretrain):
        model.train()
        for bx, bya, byb in train_loader:
            bx, bya, byb = bx.to(device, non_blocking=True), bya.to(device, non_blocking=True), byb.to(device, non_blocking=True)
            swap_mask = torch.rand(bx.size(0), device=device) > 0.5
            c1 = torch.where(swap_mask[:, None, None], byb, bya)
            c2 = torch.where(swap_mask[:, None, None], bya, byb)
            target = torch.where(swap_mask, torch.zeros(bx.size(0), device=device), torch.ones(bx.size(0), device=device))
            
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                delta, _, _ = model(bx, c1, c2)
                loss = F.binary_cross_entropy_with_logits(delta, target)
                
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        scheduler.step()
        
        model.eval()
        val_loss, val_total = 0.0, 0
        with torch.no_grad():
            for bx, bya, byb in val_loader:
                bx, bya, byb = bx.to(device, non_blocking=True), bya.to(device, non_blocking=True), byb.to(device, non_blocking=True)
                with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                    delta, _, _ = model(bx, bya, byb)
                    target = torch.ones(bx.size(0), device=device)
                    v_loss = F.binary_cross_entropy_with_logits(delta, target)
                val_loss += v_loss.item() * bx.size(0)
                val_total += bx.size(0)
        epoch_val = val_loss / max(val_total, 1)
        if epoch_val < best_val_loss:
            best_val_loss = epoch_val
            best_weights = deepcopy(model.state_dict())
            
    train_sec = (datetime.now() - t0).total_seconds()
    print(f"  -> Generic Backbone Pre-trained in {train_sec:.1f}s (Best Val Loss: {best_val_loss:.4f})")
    
    return best_weights

def calibrate_and_evaluate(base_weights, montage_channels, X_sub, YA_sub, YB_sub, K, args, device):
    """
    Calibrates on the first K trials of Subject S and evaluates strictly on the remaining (60 - K) trials.
    If K == 0, evaluates base model directly on all trials (Zero-Shot).
    """
    total_trials = len(X_sub)
    win_samples = int(TRAIN_WINDOW_SEC * FS)
    hop_samples = int(args.train_hop_sec * FS)
    
    model = CATCNDirectDecoder(
        eeg_channels=len(montage_channels),
        audio_channels=1,
        hidden_dim=64,
        max_lag_samples=8,
        dropout=0.2
    ).to(device)
    model.load_state_dict(deepcopy(base_weights))
    
    if K == 0:
        # Zero-shot evaluation on all trials
        test_x = X_sub
        test_ya = YA_sub
        test_yb = YB_sub
        res = evaluate_catcn_multiwindow_batched(model, test_x, test_ya, test_yb, device)
        return res, 0.0
        
    # Calibration set: trials 0 to K-1
    calib_trials_x = X_sub[:K]
    calib_trials_ya = YA_sub[:K]
    calib_trials_yb = YB_sub[:K]
    
    # Test set: strictly remaining trials K to 59
    test_x = X_sub[K:]
    test_ya = YA_sub[K:]
    test_yb = YB_sub[K:]
    
    # Pre-chunk calibration data
    calib_c_x, calib_c_ya, calib_c_yb = [], [], []
    for i in range(K):
        x = calib_trials_x[i]
        ya = calib_trials_ya[i]
        yb = calib_trials_yb[i]
        t_len = x.shape[1]
        start = 0
        while start + win_samples <= t_len:
            end = start + win_samples
            calib_c_x.append(x[:, start:end])
            calib_c_ya.append(ya[:, start:end])
            calib_c_yb.append(yb[:, start:end])
            start += hop_samples
            
    X_calib_t = torch.from_numpy(np.stack(calib_c_x, axis=0))
    YA_calib_t = torch.from_numpy(np.stack(calib_c_ya, axis=0))
    YB_calib_t = torch.from_numpy(np.stack(calib_c_yb, axis=0))
    
    calib_ds = TensorDataset(X_calib_t, YA_calib_t, YB_calib_t)
    calib_loader = DataLoader(calib_ds, batch_size=min(args.batch_size, len(calib_ds)), shuffle=True, pin_memory=True, num_workers=0)
    
    # Configure adaptation mode (Spatial vs Full)
    if args.adaptation_mode == "spatial":
        # Freeze temporal TCN feature extractors
        for p in model.audio_encoder.parameters():
            p.requires_grad = False
        for p in model.eeg_encoder.blocks.parameters():
            p.requires_grad = False
            
        # Adapt spatial channel mixing & classifier head
        for p in model.eeg_encoder.spatial_proj.parameters():
            p.requires_grad = True
        for p in model.eeg_encoder.bn_spatial.parameters():
            p.requires_grad = True
        for p in model.classifier_head.parameters():
            p.requires_grad = True
            
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(trainable_params, lr=args.lr_calibrate_spatial, weight_decay=1e-4)
    else:
        # Full model fine-tuning with conservative learning rate
        optimizer = optim.Adam(model.parameters(), lr=args.lr_calibrate_full, weight_decay=1e-4)
        
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    t0 = datetime.now()
    for epoch in range(args.epochs_calibrate):
        model.train()
        for bx, bya, byb in calib_loader:
            bx, bya, byb = bx.to(device, non_blocking=True), bya.to(device, non_blocking=True), byb.to(device, non_blocking=True)
            swap_mask = torch.rand(bx.size(0), device=device) > 0.5
            c1 = torch.where(swap_mask[:, None, None], byb, bya)
            c2 = torch.where(swap_mask[:, None, None], bya, byb)
            target = torch.where(swap_mask, torch.zeros(bx.size(0), device=device), torch.ones(bx.size(0), device=device))
            
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                delta, _, _ = model(bx, c1, c2)
                loss = F.binary_cross_entropy_with_logits(delta, target)
                
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                
    calib_sec = (datetime.now() - t0).total_seconds()
    
    # Evaluate fine-tuned model on strictly unseen test trials
    res = evaluate_catcn_multiwindow_batched(model, test_x, test_ya, test_yb, device)
    
    del model, optimizer, calib_loader, calib_ds, X_calib_t, YA_calib_t, YB_calib_t
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return res, calib_sec

def run_calibration_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_paths = subject_files()
    if not all_paths:
        print("No subjects found.")
        return
        
    target_sub = args.subject
    held_out_path = None
    train_paths = []
    for p in all_paths:
        if p.stem == target_sub:
            held_out_path = p
        else:
            train_paths.append(p)
            
    if held_out_path is None:
        print(f"Error: Subject {target_sub} not found.")
        return
        
    out_dir = Path(__file__).resolve().parents[3] / "results" / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(__file__).resolve().parents[3] / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    suite_montages = args.montage if args.montage else ["standard_64", "near_ear_expanded"]
    calibration_levels = sorted([int(k) for k in args.calibration_levels])
    
    print("\n" + "="*88)
    print(" CA-TCN FEW-SHOT PERSONALIZATION & CALIBRATION CURVE EXPERIMENT")
    print(f" Target Evaluation Subject: {target_sub}")
    print(f" Calibration Levels (K trials): {calibration_levels}")
    print(f" Montages: {suite_montages}")
    print(f" Adaptation Mode: {args.adaptation_mode.upper()} (Pretrain epochs: {args.epochs_pretrain}, Calib epochs: {args.epochs_calibrate})")
    print(f" Device: {device} | Batch Size: {args.batch_size}")
    print("="*88 + "\n")
    
    mapping, envelopes = get_mapping_data("gammatone")
    
    # 1. Load target subject data once (64 channels)
    print(f"[Stage 1]: Loading Target Subject {target_sub}...")
    target_exs = list(load_subject_examples(held_out_path))
    X_target_64, YA_target, YB_target = prepare_dataset(
        target_exs, list(range(64)), args.lowcut, args.highcut, target_sub, mapping, envelopes
    )
    YA_target = [ya.mean(axis=0, keepdims=True).astype(np.float32) if ya.shape[0] > 1 else ya.astype(np.float32) for ya in YA_target]
    YB_target = [yb.mean(axis=0, keepdims=True).astype(np.float32) if yb.shape[0] > 1 else yb.astype(np.float32) for yb in YB_target]
    print(f"  * Total Trials for {target_sub}: {len(X_target_64)}\n")
    
    curve_data = {m: {} for m in suite_montages}
    
    for m_name in suite_montages:
        print(f"\n{'#'*88}")
        print(f" MONTAGE: {m_name}")
        print(f"{'#'*88}")
        
        montage_channels = list(range(64)) if m_name == "standard_64" else MONTAGES[m_name]
        X_target_m = [x[montage_channels, :] for x in X_target_64]
        
        # Check if generic backbone checkpoint already exists
        ckpt_path = ckpt_dir / f"generic_{m_name}_heldout_{target_sub}.pt"
        if ckpt_path.exists() and not args.force_retrain:
            print(f"  -> Found cached generic backbone checkpoint: {ckpt_path.name}")
            base_weights = torch.load(ckpt_path, map_location=device)
        else:
            print(f"  -> Pre-training generic backbone on 17 training subjects...")
            base_weights = train_generic_backbone(train_paths, montage_channels, args, device, mapping, envelopes)
            torch.save(base_weights, ckpt_path)
            print(f"  -> Saved backbone checkpoint to {ckpt_path.name}")
            
        # Calibration sweep across K
        print(f"\n  -> Sweeping Calibration Levels K in {calibration_levels}...")
        for K in calibration_levels:
            res, calib_sec = calibrate_and_evaluate(
                base_weights, montage_channels, X_target_m, YA_target, YB_target, K, args, device
            )
            curve_data[m_name][K] = res
            
            n_eval_trials = len(X_target_m) if K == 0 else len(X_target_m) - K
            print(f"     K = {K:>2} ({K:>2} min calib | {n_eval_trials:>2} unseen test trials): 10s: {res[10]*100:5.1f}% | 20s: {res[20]*100:5.1f}% | 40s: {res[40]*100:5.1f}% (Adapted in {calib_sec:.2f}s)")
            
    # -------------------------------------------------------------
    # CALIBRATION-PERFORMANCE CURVE SYNTHESIS TABLE
    # -------------------------------------------------------------
    print("\n" + "="*96)
    print(f" CA-TCN CALIBRATION-PERFORMANCE CURVE ({target_sub} | 40s DECISION WINDOW)")
    print("="*96)
    print(f"{'Calibration (K)':<18} {'Duration':<12} {'64ch Ref':>12} {'Near-Ear 8ch':>15} {'Near-Ear Gap':>16} {'Gain(8ch vs Zero-Shot)':>24}")
    print("-" * 96)
    
    csv_rows = ["K_Trials,Duration_Min,Ref_64ch_40s,NearEar_8ch_40s,NearEar_Gap_pp,Personalization_Gain_8ch_pp"]
    
    has_64 = "standard_64" in curve_data
    has_near = "near_ear_expanded" in curve_data
    
    zero_shot_near = curve_data["near_ear_expanded"][0][40] * 100 if has_near and 0 in curve_data["near_ear_expanded"] else np.nan
    
    for K in calibration_levels:
        dur_str = f"{K} min" if K > 0 else "0 min (Zero-Shot)"
        k_str = f"K = {K}" if K > 0 else "K = 0 (Zero-Shot)"
        
        acc_64 = curve_data["standard_64"][K][40] * 100 if has_64 and K in curve_data["standard_64"] else np.nan
        acc_near = curve_data["near_ear_expanded"][K][40] * 100 if has_near and K in curve_data["near_ear_expanded"] else np.nan
        
        gap = acc_64 - acc_near if (not np.isnan(acc_64) and not np.isnan(acc_near)) else np.nan
        gain_8ch = acc_near - zero_shot_near if (not np.isnan(acc_near) and not np.isnan(zero_shot_near)) else np.nan
        
        str_64 = f"{acc_64:.1f}%" if not np.isnan(acc_64) else "N/A"
        str_near = f"{acc_near:.1f}%" if not np.isnan(acc_near) else "N/A"
        str_gap = f"{gap:+.1f} pp" if not np.isnan(gap) else "N/A"
        str_gain = f"{gain_8ch:+.1f} pp" if not np.isnan(gain_8ch) else "N/A"
        
        print(f"{k_str:<18} {dur_str:<12} {str_64:>12} {str_near:>15} {str_gap:>16} {str_gain:>24}")
        csv_rows.append(f"{K},{K},{acc_64:.2f},{acc_near:.2f},{gap:.2f},{gain_8ch:.2f}")
        
    print("="*96)
    
    # Save CSV
    out_file = out_dir / f"calibration_curve_{target_sub}.csv"
    with open(out_file, "w") as f:
        f.write("\n".join(csv_rows))
        
    print(f"\nCalibration artifact saved to: {out_file}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CA-TCN Few-Shot Personalization & Calibration Curve Runner")
    parser.add_argument("--subject", type=str, default="S1_data_preproc", help="Subject to evaluate")
    parser.add_argument("--montage", type=str, nargs="+", default=["standard_64", "near_ear_expanded"], help="Montages to evaluate")
    parser.add_argument("--calibration-levels", type=int, nargs="+", default=[0, 1, 2, 5, 10, 20], help="Calibration trial levels K")
    parser.add_argument("--adaptation-mode", type=str, choices=["spatial", "full"], default="spatial", help="Fine-tuning strategy")
    parser.add_argument("--epochs-pretrain", type=int, default=15, help="Epochs for generic backbone pre-training")
    parser.add_argument("--epochs-calibrate", type=int, default=5, help="Epochs for few-shot user calibration")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train_hop_sec", type=float, default=2.5)
    parser.add_argument("--lowcut", type=float, default=1.0)
    parser.add_argument("--highcut", type=float, default=6.0)
    parser.add_argument("--lr-pretrain", type=float, default=2e-4)
    parser.add_argument("--lr-calibrate-spatial", type=float, default=1e-4)
    parser.add_argument("--lr-calibrate-full", type=float, default=5e-5)
    parser.add_argument("--force-retrain", action="store_true", help="Force retraining base backbone even if cached")
    args = parser.parse_args()
    
    run_calibration_experiment(args)
