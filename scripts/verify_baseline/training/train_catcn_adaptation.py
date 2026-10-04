import argparse
import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

import json
import time
import math
from pathlib import Path
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
VERIFY_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(VERIFY_ROOT) not in sys.path:
    sys.path.insert(0, str(VERIFY_ROOT))

from models.catcn import CATCNDirectDecoder
from src.streaming.causal_filters import StreamingCausalEEGFilter
from training.montages import MONTAGES, DTU_CHANNELS
from training.train_matchnet_wavlm import get_mapping_data, prepare_dataset, FS
from baselines.ridge_aad import load_subject_examples, subject_files

def butter_lowpass_sosfilt(data: np.ndarray, cutoff: float, fs: float, order: int = 2) -> np.ndarray:
    from scipy import signal
    sos = signal.butter(order, cutoff, btype='low', fs=fs, output='sos')
    return signal.sosfilt(sos, data).astype(np.float32)

@torch.no_grad()
def evaluate_windows(model, eeg_list, ya_list, yb_list, window_sec, fs, device):
    """
    Evaluates CA-TCN on non-overlapping windows across a set of trials.
    """
    model.eval()
    window_samples = int(window_sec * fs)
    deltas = []
    
    for eeg, ya, yb in zip(eeg_list, ya_list, yb_list):
        t_len = min(len(eeg), len(ya), len(yb))
        for s in range(0, t_len - window_samples + 1, window_samples):
            e = s + window_samples
            w_e = torch.from_numpy(eeg[s:e].T.copy()).unsqueeze(0).float().to(device)
            w_a = torch.from_numpy(ya[s:e].copy()).unsqueeze(0).unsqueeze(0).float().to(device)
            w_b = torch.from_numpy(yb[s:e].copy()).unsqueeze(0).unsqueeze(0).float().to(device)
            
            d, _, _ = model(w_e, w_a, w_b)
            deltas.append(d.item())
            
    deltas = np.array(deltas)
    if len(deltas) == 0:
        return 50.0, 0, 0, deltas
    correct = np.sum(deltas > 0)
    total = len(deltas)
    acc = (correct / total) * 100.0
    return acc, int(correct), int(total), deltas

@torch.no_grad()
def evaluate_full_trial_decisions(model, eeg_list, ya_list, yb_list, window_sec, step_sec, fs, device):
    """
    Evaluates rolling 5.0s window streaming decisions across all trials.
    Computes Trial Majority, Cumulative Margin, Mean Lock Time, and Neural Margin.
    """
    model.eval()
    window_samples = int(window_sec * fs)
    step_samples = int(step_sec * fs)
    
    trial_majority_correct = 0
    trial_cumulative_correct = 0
    all_time_lock_pcts = []
    all_mean_deltas = []
    
    for eeg, ya, yb in zip(eeg_list, ya_list, yb_list):
        t_len = min(len(eeg), len(ya), len(yb))
        trial_deltas = []
        
        curr_start = 0
        while curr_start + window_samples <= t_len:
            curr_end = curr_start + window_samples
            w_e = eeg[curr_start:curr_end]
            w_a = ya[curr_start:curr_end]
            w_b = yb[curr_start:curr_end]
            
            # Causal window standardization (matching streaming ring buffer)
            w_e = (w_e - np.mean(w_e, axis=0, keepdims=True)) / (np.std(w_e, axis=0, keepdims=True) + 1e-8)
            w_a = (w_a - np.mean(w_a)) / (np.std(w_a) + 1e-8)
            w_b = (w_b - np.mean(w_b)) / (np.std(w_b) + 1e-8)
            
            t_e = torch.from_numpy(w_e.T.copy()).unsqueeze(0).float().to(device)
            t_a = torch.from_numpy(w_a.copy()).unsqueeze(0).unsqueeze(0).float().to(device)
            t_b = torch.from_numpy(w_b.copy()).unsqueeze(0).unsqueeze(0).float().to(device)
            
            d, _, _ = model(t_e, t_a, t_b)
            trial_deltas.append(d.item())
            curr_start += step_samples
            
        if trial_deltas:
            trial_deltas = np.array(trial_deltas)
            correct_steps = np.sum(trial_deltas > 0)
            maj_win = correct_steps >= (len(trial_deltas) / 2.0)
            if maj_win:
                trial_majority_correct += 1
                
            cum_win = np.sum(trial_deltas) > 0
            if cum_win:
                trial_cumulative_correct += 1
                
            time_lock = (correct_steps / len(trial_deltas)) * 100.0
            all_time_lock_pcts.append(time_lock)
            all_mean_deltas.append(np.mean(trial_deltas))
            
    n_trials = len(eeg_list)
    maj_acc = (trial_majority_correct / max(1, n_trials)) * 100.0
    cum_acc = (trial_cumulative_correct / max(1, n_trials)) * 100.0
    mean_lock = np.mean(all_time_lock_pcts) if all_time_lock_pcts else 50.0
    mean_margin = np.mean(all_mean_deltas) if all_mean_deltas else 0.0
    
    return {
        "majority_correct": trial_majority_correct,
        "cumulative_correct": trial_cumulative_correct,
        "total_trials": n_trials,
        "majority_acc": maj_acc,
        "cumulative_acc": cum_acc,
        "mean_lock_pct": mean_lock,
        "mean_margin": mean_margin
    }

def run_evaluation_suite(model, eeg_list, ya_list, yb_list, fs, device):
    acc_5s, c_5s, n_5s, _ = evaluate_windows(model, eeg_list, ya_list, yb_list, 5.0, fs, device)
    acc_10s, c_10s, n_10s, _ = evaluate_windows(model, eeg_list, ya_list, yb_list, 10.0, fs, device)
    acc_20s, c_20s, n_20s, _ = evaluate_windows(model, eeg_list, ya_list, yb_list, 20.0, fs, device)
    trial_metrics = evaluate_full_trial_decisions(model, eeg_list, ya_list, yb_list, 5.0, 0.5, fs, device)
    
    return {
        "acc_5s": acc_5s,
        "acc_10s": acc_10s,
        "acc_20s": acc_20s,
        "majority_acc": trial_metrics["majority_acc"],
        "cumulative_acc": trial_metrics["cumulative_acc"],
        "mean_lock_pct": trial_metrics["mean_lock_pct"],
        "mean_margin": trial_metrics["mean_margin"],
        "total_trials": trial_metrics["total_trials"]
    }

def process_subject_trials(examples, montage_channels, sub_id, mapping, envelopes, causal_filter, fs):
    """
    Extracts, causal-filters, and standardizes continuous trials for a subject.
    Returns: list of eeg arrays, list of ya arrays, list of yb arrays.
    """
    _, ya_raw, yb_raw = prepare_dataset(examples, montage_channels, 1.0, 6.0, sub_id, mapping, envelopes)
    ya_clean = [ya.mean(axis=0, keepdims=True).squeeze(0).astype(np.float32) if ya.shape[0] > 1 else ya.squeeze(0).astype(np.float32) for ya in ya_raw]
    yb_clean = [yb.mean(axis=0, keepdims=True).squeeze(0).astype(np.float32) if yb.shape[0] > 1 else yb.squeeze(0).astype(np.float32) for yb in yb_raw]
    
    eeg_out, ya_out, yb_out = [], [], []
    for idx in range(min(len(examples), len(ya_clean))):
        raw_eeg = examples[idx].eeg[:, montage_channels].astype(np.float32)
        min_len = min(len(raw_eeg), len(ya_clean[idx]), len(yb_clean[idx]))
        raw_eeg = raw_eeg[:min_len]
        
        causal_filter.reset()
        eeg_c = causal_filter.process_chunk(raw_eeg)
        eeg_c = (eeg_c - np.mean(eeg_c, axis=0, keepdims=True)) / (np.std(eeg_c, axis=0, keepdims=True) + 1e-8)
        
        ya_c = butter_lowpass_sosfilt(ya_clean[idx][:min_len], 8.0, fs, order=2).astype(np.float32)
        yb_c = butter_lowpass_sosfilt(yb_clean[idx][:min_len], 8.0, fs, order=2).astype(np.float32)
        ya_c = (ya_c - np.mean(ya_c)) / (np.std(ya_c) + 1e-8)
        yb_c = (yb_c - np.mean(yb_c)) / (np.std(yb_c) + 1e-8)
        
        eeg_out.append(eeg_c)
        ya_out.append(ya_c)
        yb_out.append(yb_c)
        
    return eeg_out, ya_out, yb_out

def chunk_trials(eeg_list, ya_list, yb_list, win_sec, hop_sec, fs):
    win_samples = int(win_sec * fs)
    hop_samples = int(hop_sec * fs)
    x_chunks, ya_chunks, yb_chunks = [], [], []
    
    for eeg, ya, yb in zip(eeg_list, ya_list, yb_list):
        t_len = min(len(eeg), len(ya), len(yb))
        x_t = eeg.T
        ya_t = np.expand_dims(ya, axis=0)
        yb_t = np.expand_dims(yb, axis=0)
        
        start = 0
        while start + win_samples <= t_len:
            end = start + win_samples
            x_chunks.append(x_t[:, start:end])
            ya_chunks.append(ya_t[:, start:end])
            yb_chunks.append(yb_t[:, start:end])
            start += hop_samples
            
    return np.stack(x_chunks, axis=0), np.stack(ya_chunks, axis=0), np.stack(yb_chunks, axis=0)

def train_backbone(train_paths, montage_channels, mapping, envelopes, causal_filter, args, device):
    """
    Trains the 17-subject Universal CA-TCN backbone.
    """
    print(f"\n  [STAGE 1] Pre-training Universal CA-TCN Backbone on {len(train_paths)} subjects...")
    t0 = time.time()
    all_x, all_ya, all_yb = [], [], []
    
    for p in train_paths:
        sub_id = p.stem.split("_")[0]
        exs = list(load_subject_examples(p))
        eeg_l, ya_l, yb_l = process_subject_trials(exs, montage_channels, sub_id, mapping, envelopes, causal_filter, FS)
        cx, cya, cyb = chunk_trials(eeg_l, ya_l, yb_l, args.window_sec, args.hop_sec, FS)
        all_x.append(cx)
        all_ya.append(cya)
        all_yb.append(cyb)
        
    X_tr = np.concatenate(all_x, axis=0)
    YA_tr = np.concatenate(all_ya, axis=0)
    YB_tr = np.concatenate(all_yb, axis=0)
    print(f"  [DATA] Prepared {len(X_tr)} universal training chunks in {time.time() - t0:.1f}s.")
    
    model = CATCNDirectDecoder(eeg_channels=len(montage_channels), audio_channels=1, hidden_dim=args.hidden_dim, max_lag_samples=8).to(device)
    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(YA_tr), torch.from_numpy(YB_tr))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2 if os.name != 'nt' else 0)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_pretrain, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
    
    for epoch in range(1, args.epochs_pretrain + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for bx, bya, byb in train_loader:
            bx, bya, byb = bx.to(device, non_blocking=True), bya.to(device, non_blocking=True), byb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                delta, (la, lb), _ = model(bx, bya, byb)
                loss = torch.clamp(0.5 - (la - lb), min=0.0).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()
        if epoch % 2 == 0 or epoch == args.epochs_pretrain:
            print(f"    * Backbone Epoch {epoch:02d}/{args.epochs_pretrain:02d} | Loss: {total_loss/max(1, n_batches):.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
            
    print(f"  [BACKBONE] Universal training finished in {(time.time() - t0)/60.0:.1f} minutes.")
    return model

def main():
    parser = argparse.ArgumentParser(description="CA-TCN Few-Shot Subject Adaptation & Personalization Benchmark")
    parser.add_argument("--subject", type=str, default="S6", help="Target subject (e.g. S6, S11, S1)")
    parser.add_argument("--calib_trials", type=int, default=12, help="Number of calibration trials (trials 0 to K-1)")
    parser.add_argument("--montage", type=str, default="near_ear_expanded", help="Electrode montage")
    parser.add_argument("--window_sec", type=float, default=5.0, help="Window size in seconds")
    parser.add_argument("--hop_sec", type=float, default=2.5, help="Training hop size in seconds")
    parser.add_argument("--epochs_pretrain", type=int, default=8, help="Backbone pretraining epochs")
    parser.add_argument("--epochs_calib", type=int, default=10, help="Few-shot calibration epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Backbone learning rate")
    parser.add_argument("--lr_calib_full", type=float, default=1e-4, help="Learning rate for full fine-tuning")
    parser.add_argument("--lr_calib_spatial", type=float, default=2e-4, help="Learning rate for spatial-only adaptation")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--backbone_path", type=str, default="", help="Path to pre-trained checkpoint to adapt")
    parser.add_argument("--force_retrain_backbone", action="store_true", help="Force retraining universal backbone")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 96)
    print(f"  CA-TCN FEW-SHOT ADAPTATION BENCHMARK: Target Subject = {args.subject}")
    print(f"  Calibration Trials: {args.calib_trials} (Trials 00-{args.calib_trials-1:02d}) | Held-Out Test Trials: {60-args.calib_trials} (Trials {args.calib_trials:02d}-59)")
    print(f"  Device: {device} | Montage: {args.montage} (8 ch) | Hidden Dim: {args.hidden_dim}")
    print("=" * 96)
    
    montage_channels = MONTAGES[args.montage]
    all_paths = subject_files()
    if not all_paths:
        print("[ERROR] No DTU subject files found. Mount dataset first.")
        return
        
    mapping, envelopes = get_mapping_data("gammatone")
    target_path = next((p for p in all_paths if p.stem.split("_")[0] == args.subject), None)
    if not target_path:
        print(f"[ERROR] Target subject {args.subject} not found in dataset.")
        return
        
    train_paths = [p for p in all_paths if p.stem.split("_")[0] != args.subject]
    causal_filter = StreamingCausalEEGFilter(fs=FS, lowcut=1.0, highcut=6.0, order=2, n_channels=len(montage_channels))
    
    # 1. Obtain Universal Backbone
    ckpt_dir = Path("/kaggle/working/checkpoints") if Path("/kaggle/working").exists() else Path("checkpoints/adaptation")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    backbone_ckpt = Path(args.backbone_path) if args.backbone_path and Path(args.backbone_path).exists() else (ckpt_dir / f"catcn_univ_heldout_{args.subject}.pt")
    
    if backbone_ckpt.exists() and not args.force_retrain_backbone:
        print(f"  [CHECKPOINT] Loading backbone from: {backbone_ckpt}")
        try:
            raw_sd = torch.load(backbone_ckpt, map_location=device, weights_only=False)
        except TypeError:
            raw_sd = torch.load(backbone_ckpt, map_location=device)
        if "model_state_dict" in raw_sd:
            univ_model.load_state_dict(raw_sd["model_state_dict"])
        elif "model" in raw_sd:
            univ_model.load_state_dict(raw_sd["model"])
        else:
            univ_model.load_state_dict(raw_sd)
    else:
        univ_model = train_backbone(train_paths, montage_channels, mapping, envelopes, causal_filter, args, device)
        torch.save(univ_model.state_dict(), backbone_ckpt)
        print(f"  [CHECKPOINT] Saved universal backbone to {backbone_ckpt.name}")
        
    # 2. Prepare Target Subject Data & Partition
    print(f"\n  [STAGE 2] Preparing Target Subject {args.subject} data partition...")
    target_exs = list(load_subject_examples(target_path))
    eeg_all, ya_all, yb_all = process_subject_trials(target_exs, montage_channels, args.subject, mapping, envelopes, causal_filter, FS)
    
    # Calibration set: Trials 0 to K-1
    K = args.calib_trials
    eeg_calib, ya_calib, yb_calib = eeg_all[:K], ya_all[:K], yb_all[:K]
    # Sequestered Test set: Trials K to 59
    eeg_test, ya_test, yb_test = eeg_all[K:], ya_all[K:], yb_all[K:]
    print(f"  [PARTITION] Calibration Set: {len(eeg_calib)} trials ({K} min) | Sequestered Test Set: {len(eeg_test)} trials ({len(eeg_test)} min)")
    
    # Chunk calibration data
    X_calib, YA_calib, YB_calib = chunk_trials(eeg_calib, ya_calib, yb_calib, args.window_sec, args.hop_sec, FS)
    print(f"  [DATA] Generated {len(X_calib)} calibration chunks from trials 00-{K-1:02d}.")
    calib_ds = TensorDataset(torch.from_numpy(X_calib), torch.from_numpy(YA_calib), torch.from_numpy(YB_calib))
    calib_loader = DataLoader(calib_ds, batch_size=min(args.batch_size, len(calib_ds)), shuffle=True)
    
    results = {}
    
    # --------------------------------------------------------------------------
    # REGIME 1: Zero-Shot Universal (Baseline)
    # --------------------------------------------------------------------------
    print("\n" + "-" * 90)
    print("  [REGIME 1] Evaluating Zero-Shot Universal CA-TCN (0 calibration trials)...")
    res_zero = run_evaluation_suite(univ_model, eeg_test, ya_test, yb_test, FS, device)
    results["1. Zero-Shot Universal"] = res_zero
    print(f"    * 5.0s: {res_zero['acc_5s']:.1f}% | 10.0s: {res_zero['acc_10s']:.1f}% | 20.0s: {res_zero['acc_20s']:.1f}% | Majority: {res_zero['majority_acc']:.1f}% | Margin: {res_zero['mean_margin']:+.2f}")
    
    # --------------------------------------------------------------------------
    # REGIME 2: From-Scratch Personalized (12 Calibration Trials Only)
    # --------------------------------------------------------------------------
    print("\n" + "-" * 90)
    print(f"  [REGIME 2] Training From Scratch purely on {K} calibration trials (Workspace Rule 3 test)...")
    torch.manual_seed(42)
    scratch_model = CATCNDirectDecoder(eeg_channels=len(montage_channels), audio_channels=1, hidden_dim=args.hidden_dim, max_lag_samples=8).to(device)
    opt_scratch = optim.AdamW(scratch_model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched_scratch = optim.lr_scheduler.CosineAnnealingLR(opt_scratch, T_max=args.epochs_calib * 2, eta_min=1e-5)
    
    scratch_model.train()
    for ep in range(1, args.epochs_calib * 2 + 1):
        for bx, bya, byb in calib_loader:
            bx, bya, byb = bx.to(device), bya.to(device), byb.to(device)
            opt_scratch.zero_grad(set_to_none=True)
            delta, (la, lb), _ = scratch_model(bx, bya, byb)
            loss = torch.clamp(0.5 - (la - lb), min=0.0).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(scratch_model.parameters(), max_norm=1.0)
            opt_scratch.step()
        sched_scratch.step()
        
    res_scratch = run_evaluation_suite(scratch_model, eeg_test, ya_test, yb_test, FS, device)
    results["2. From-Scratch (12 Trials)"] = res_scratch
    print(f"    * 5.0s: {res_scratch['acc_5s']:.1f}% | 10.0s: {res_scratch['acc_10s']:.1f}% | 20.0s: {res_scratch['acc_20s']:.1f}% | Majority: {res_scratch['majority_acc']:.1f}% | Margin: {res_scratch['mean_margin']:+.2f}")
    
    # --------------------------------------------------------------------------
    # REGIME 3: Full Backbone Fine-Tuning
    # --------------------------------------------------------------------------
    print("\n" + "-" * 90)
    print(f"  [REGIME 3] Full-Backbone Fine-Tuning (All 38k parameters, LR = {args.lr_calib_full})...")
    full_model = deepcopy(univ_model)
    opt_full = optim.AdamW(full_model.parameters(), lr=args.lr_calib_full, weight_decay=1e-4)
    sched_full = optim.lr_scheduler.CosineAnnealingLR(opt_full, T_max=args.epochs_calib, eta_min=1e-6)
    
    full_model.train()
    for ep in range(1, args.epochs_calib + 1):
        for bx, bya, byb in calib_loader:
            bx, bya, byb = bx.to(device), bya.to(device), byb.to(device)
            opt_full.zero_grad(set_to_none=True)
            delta, (la, lb), _ = full_model(bx, bya, byb)
            loss = torch.clamp(0.5 - (la - lb), min=0.0).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(full_model.parameters(), max_norm=1.0)
            opt_full.step()
        sched_full.step()
        
    res_full = run_evaluation_suite(full_model, eeg_test, ya_test, yb_test, FS, device)
    results["3. Full Fine-Tuning"] = res_full
    print(f"    * 5.0s: {res_full['acc_5s']:.1f}% | 10.0s: {res_full['acc_10s']:.1f}% | 20.0s: {res_full['acc_20s']:.1f}% | Majority: {res_full['majority_acc']:.1f}% | Margin: {res_full['mean_margin']:+.2f}")
    
    # --------------------------------------------------------------------------
    # REGIME 4: Spatial & BN-Only Adaptation (Recommended)
    # --------------------------------------------------------------------------
    print("\n" + "-" * 90)
    print(f"  [REGIME 4] Spatial & BN-Only Adaptation (Temporal TCN Frozen, Adapt 640 params, LR = {args.lr_calib_spatial})...")
    spatial_model = deepcopy(univ_model)
    
    # Freeze all temporal TCN blocks
    for p in spatial_model.audio_encoder.parameters():
        p.requires_grad = False
    for p in spatial_model.eeg_encoder.blocks.parameters():
        p.requires_grad = False
    for p in spatial_model.classifier_head.parameters():
        p.requires_grad = False
        
    # Unfreeze spatial projection and batch norms
    for p in spatial_model.eeg_encoder.spatial_proj.parameters():
        p.requires_grad = True
    for p in spatial_model.eeg_encoder.bn_spatial.parameters():
        p.requires_grad = True
        
    trainable_params = [p for p in spatial_model.parameters() if p.requires_grad]
    param_count = sum(p.numel() for p in trainable_params)
    print(f"    -> Active Trainable Parameters: {param_count:,} / {sum(p.numel() for p in spatial_model.parameters()):,}")
    
    opt_spatial = optim.AdamW(trainable_params, lr=args.lr_calib_spatial, weight_decay=1e-4)
    sched_spatial = optim.lr_scheduler.CosineAnnealingLR(opt_spatial, T_max=args.epochs_calib, eta_min=1e-5)
    
    spatial_model.train()
    for ep in range(1, args.epochs_calib + 1):
        for bx, bya, byb in calib_loader:
            bx, bya, byb = bx.to(device), bya.to(device), byb.to(device)
            opt_spatial.zero_grad(set_to_none=True)
            delta, (la, lb), _ = spatial_model(bx, bya, byb)
            loss = torch.clamp(0.5 - (la - lb), min=0.0).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            opt_spatial.step()
        sched_spatial.step()
        
    res_spatial = run_evaluation_suite(spatial_model, eeg_test, ya_test, yb_test, FS, device)
    results["4. Spatial & BN-Only (Ours)"] = res_spatial
    print(f"    * 5.0s: {res_spatial['acc_5s']:.1f}% | 10.0s: {res_spatial['acc_10s']:.1f}% | 20.0s: {res_spatial['acc_20s']:.1f}% | Majority: {res_spatial['majority_acc']:.1f}% | Margin: {res_spatial['mean_margin']:+.2f}")
    
    # Save adapted spatial model checkpoint
    adapted_dir = Path("/kaggle/working/loso_checkpoints") if Path("/kaggle/working").exists() else Path("checkpoints/adapted")
    adapted_dir.mkdir(parents=True, exist_ok=True)
    adapted_path = adapted_dir / f"catcn_adapted_{args.subject}.pt"
    torch.save({
        "model_state_dict": spatial_model.state_dict(),
        "calib_trials": int(K),
        "subject": str(args.subject),
        "acc_5s": float(res_spatial['acc_5s']),
        "acc_10s": float(res_spatial['acc_10s']),
        "acc_20s": float(res_spatial['acc_20s']),
        "majority_acc": float(res_spatial['majority_acc'])
    }, adapted_path)
    print(f"  [CHECKPOINT] Saved adapted model (Spatial & BN adapted) to: {adapted_path}")
    
    # --------------------------------------------------------------------------
    # SYNTHESIS COMPARISON TABLE
    # --------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print(f"  CA-TCN FEW-SHOT ADAPTATION BENCHMARK: SUBJECT {args.subject}")
    print(f"  Evaluated Strictly on {len(eeg_test)} Sequestered Test Trials (Trials {K:02d}-59, 100% Unseen)")
    print("=" * 115)
    print(f"  {'Adaptation Regime':<28} | {'5.0s 2AFC':<10} | {'10.0s 2AFC':<10} | {'20.0s 2AFC':<10} | {'Majority Win':<14} | {'Cumul Win':<10} | {'Mean Margin'}")
    print("  " + "-" * 111)
    
    for name, r in results.items():
        print(f"  {name:<28} | {r['acc_5s']:>8.1f}% | {r['acc_10s']:>8.1f}% | {r['acc_20s']:>8.1f}% | {r['majority_acc']:>12.1f}% | {r['cumulative_acc']:>8.1f}% | {r['mean_margin']:>+6.2f}")
        
    print("=" * 115)
    
    # Delta Gains compared to Zero-Shot
    gain_scratch = results["2. From-Scratch (12 Trials)"]["majority_acc"] - res_zero["majority_acc"]
    gain_full = results["3. Full Fine-Tuning"]["majority_acc"] - res_zero["majority_acc"]
    gain_spatial = results["4. Spatial & BN-Only (Ours)"]["majority_acc"] - res_zero["majority_acc"]
    print(f"  Personalization Gain vs Zero-Shot (Majority Win %):")
    print(f"    - From-Scratch (12 Trials):   {gain_scratch:>+5.1f} pp")
    print(f"    - Full Fine-Tuning:          {gain_full:>+5.1f} pp")
    print(f"    - Spatial & BN Adaptation:   {gain_spatial:>+5.1f} pp")
    print("=" * 115 + "\n")

if __name__ == "__main__":
    main()
