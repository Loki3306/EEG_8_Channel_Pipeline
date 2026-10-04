import argparse
import sys
import os
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

def resolve_path(target_path_str: str) -> Path:
    p = Path(target_path_str)
    if p.exists():
        return p
    for root in ["/kaggle/working", "/kaggle/input", "."]:
        cand = Path(root) / target_path_str
        if cand.exists():
            return cand
        cand_name = Path(root) / p.name
        if cand_name.exists():
            return cand_name
    return p

def butter_lowpass_sosfilt(data: np.ndarray, cutoff: float, fs: float, order: int = 2) -> np.ndarray:
    from scipy import signal
    sos = signal.butter(order, cutoff, btype='low', fs=fs, output='sos')
    return signal.sosfilt(sos, data).astype(np.float32)

@torch.no_grad()
def evaluate_loso_fold(model, eeg_list, ya_list, yb_list, window_sec, fs, device):
    """
    Evaluates CA-TCN on non-overlapping windows for a completely held-out test subject.
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
    Evaluates rolling 5.0s window streaming decisions across all trials for a held-out subject.
    Computes Trial Majority and Cumulative Margin decisions.
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
            # Majority winner: more than half steps positive (Stream A)
            correct_steps = np.sum(trial_deltas > 0)
            maj_win = correct_steps >= (len(trial_deltas) / 2.0)
            if maj_win:
                trial_majority_correct += 1
                
            mean_d = float(np.mean(trial_deltas))
            all_mean_deltas.append(mean_d)
            if mean_d > 0:
                trial_cumulative_correct += 1
                
            lock_pct = (correct_steps / len(trial_deltas)) * 100.0
            all_time_lock_pcts.append(lock_pct)
            
    n_trials = len(eeg_list)
    maj_acc = (trial_majority_correct / max(1, n_trials)) * 100.0
    cum_acc = (trial_cumulative_correct / max(1, n_trials)) * 100.0
    mean_lock = float(np.mean(all_time_lock_pcts)) if all_time_lock_pcts else 50.0
    grand_margin = float(np.mean(all_mean_deltas)) if all_mean_deltas else 0.0
    
    return {
        "majority_acc": maj_acc,
        "majority_correct": trial_majority_correct,
        "cumulative_acc": cum_acc,
        "cumulative_correct": trial_cumulative_correct,
        "mean_lock_pct": mean_lock,
        "mean_margin": grand_margin,
        "total_trials": n_trials
    }

def train_loso_fold(
    test_sub: str,
    all_paths: list,
    montage_channels: list,
    mapping: dict,
    envelopes: dict,
    args,
    device
):
    """
    Trains CA-TCN on 17 subjects and evaluates strictly on the held-out 18th subject.
    """
    fs = FS
    n_ch = len(montage_channels)
    
    train_paths = [p for p in all_paths if p.stem.split("_")[0] != test_sub]
    test_path = [p for p in all_paths if p.stem.split("_")[0] == test_sub]
    
    if not test_path:
        print(f"[ERROR] Could not find subject file for {test_sub}. Skipping.")
        return None
    test_path = test_path[0]
    
    print("\n" + "=" * 92)
    print(f"  LEAVE-ONE-SUBJECT-OUT (LOSO) FOLD: Held-Out Target = {test_sub}")
    print(f"  Training Subjects ({len(train_paths)}): {[p.stem.split('_')[0] for p in train_paths]}")
    print("=" * 92)
    
    causal_eeg_filter = StreamingCausalEEGFilter(lowcut=args.lowcut, highcut=args.highcut, fs=fs, order=2, n_channels=n_ch)
    win_samples = int(args.window_sec * fs)
    hop_samples = int(args.hop_sec * fs)
    
    # 1. Prepare Training Data from 17 subjects
    X_tr_list, YA_tr_list, YB_tr_list = [], [], []
    t_data_start = time.time()
    
    for p in train_paths:
        sub_name = p.stem.split("_")[0]
        exs = list(load_subject_examples(p))
        if args.smoke_test:
            exs = exs[:6] # 6 trials in smoke test mode
            
        _, YA_raw, YB_raw = prepare_dataset(exs, montage_channels, args.lowcut, args.highcut, sub_name, mapping, envelopes)
        YA_clean = [ya.mean(axis=0, keepdims=True).squeeze(0).astype(np.float32) if ya.shape[0] > 1 else ya.squeeze(0).astype(np.float32) for ya in YA_raw]
        YB_clean = [yb.mean(axis=0, keepdims=True).squeeze(0).astype(np.float32) if yb.shape[0] > 1 else yb.squeeze(0).astype(np.float32) for yb in YB_raw]
        
        n_valid = min(len(exs), len(YA_clean))
        for idx in range(n_valid):
            raw_eeg = exs[idx].eeg[:, montage_channels].astype(np.float32)
            min_len = min(len(raw_eeg), len(YA_clean[idx]), len(YB_clean[idx]))
            raw_eeg = raw_eeg[:min_len]
            
            causal_eeg_filter.reset()
            eeg_c = causal_eeg_filter.process_chunk(raw_eeg)
            eeg_c = (eeg_c - np.mean(eeg_c, axis=0, keepdims=True)) / (np.std(eeg_c, axis=0, keepdims=True) + 1e-8)
            
            ya_c = butter_lowpass_sosfilt(YA_clean[idx][:min_len], 8.0, fs, order=2).astype(np.float32)
            yb_c = butter_lowpass_sosfilt(YB_clean[idx][:min_len], 8.0, fs, order=2).astype(np.float32)
            ya_c = (ya_c - np.mean(ya_c)) / (np.std(ya_c) + 1e-8)
            yb_c = (yb_c - np.mean(yb_c)) / (np.std(yb_c) + 1e-8)
            
            x_t = eeg_c.T
            ya_t = np.expand_dims(ya_c, axis=0)
            yb_t = np.expand_dims(yb_c, axis=0)
            
            start = 0
            while start + win_samples <= min_len:
                end = start + win_samples
                X_tr_list.append(x_t[:, start:end])
                YA_tr_list.append(ya_t[:, start:end])
                YB_tr_list.append(yb_t[:, start:end])
                start += hop_samples
                
    print(f"  [DATA] Prepared {len(X_tr_list)} training chunks ({len(train_paths)} subjects) in {time.time() - t_data_start:.1f}s.")
    
    # 2. Prepare Held-Out Test Subject Data (Strictly Sequestered)
    test_exs = list(load_subject_examples(test_path))
    if args.smoke_test:
        test_exs = test_exs[:6]
    _, YA_test_raw, YB_test_raw = prepare_dataset(test_exs, montage_channels, args.lowcut, args.highcut, test_sub, mapping, envelopes)
    YA_test_clean = [ya.mean(axis=0, keepdims=True).squeeze(0).astype(np.float32) if ya.shape[0] > 1 else ya.squeeze(0).astype(np.float32) for ya in YA_test_raw]
    YB_test_clean = [yb.mean(axis=0, keepdims=True).squeeze(0).astype(np.float32) if yb.shape[0] > 1 else yb.squeeze(0).astype(np.float32) for yb in YB_test_raw]
    
    test_eeg_list, test_ya_list, test_yb_list = [], [], []
    for idx in range(min(len(test_exs), len(YA_test_clean))):
        raw_eeg = test_exs[idx].eeg[:, montage_channels].astype(np.float32)
        min_len = min(len(raw_eeg), len(YA_test_clean[idx]), len(YB_test_clean[idx]))
        raw_eeg = raw_eeg[:min_len]
        
        causal_eeg_filter.reset()
        eeg_c = causal_eeg_filter.process_chunk(raw_eeg)
        eeg_c = (eeg_c - np.mean(eeg_c, axis=0, keepdims=True)) / (np.std(eeg_c, axis=0, keepdims=True) + 1e-8)
        
        ya_c = butter_lowpass_sosfilt(YA_test_clean[idx][:min_len], 8.0, fs, order=2).astype(np.float32)
        yb_c = butter_lowpass_sosfilt(YB_test_clean[idx][:min_len], 8.0, fs, order=2).astype(np.float32)
        ya_c = (ya_c - np.mean(ya_c)) / (np.std(ya_c) + 1e-8)
        yb_c = (yb_c - np.mean(yb_c)) / (np.std(yb_c) + 1e-8)
        
        test_eeg_list.append(eeg_c)
        test_ya_list.append(ya_c)
        test_yb_list.append(yb_c)
        
    print(f"  [DATA] Sequestered {len(test_eeg_list)} held-out trials for {test_sub}.")
    
    # 3. Model Initialization (From Scratch - Workspace Rule 3)
    fold_seed = 42 + int(test_sub[1:]) if test_sub[1:].isdigit() else 42
    torch.manual_seed(fold_seed)
    np.random.seed(fold_seed)
    model = CATCNDirectDecoder(eeg_channels=n_ch, audio_channels=1, hidden_dim=args.hidden_dim, max_lag_samples=8).to(device)
    
    train_ds = TensorDataset(
        torch.from_numpy(np.stack(X_tr_list, axis=0)),
        torch.from_numpy(np.stack(YA_tr_list, axis=0)),
        torch.from_numpy(np.stack(YB_tr_list, axis=0))
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2 if os.name != 'nt' else 0)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 4. Training Loop
    epochs = args.epochs if not args.smoke_test else 2
    print(f"  [TRAIN] Training CA-TCN for {epochs} epochs on {len(X_tr_list)} chunks...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
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
        if epoch % 2 == 0 or epoch == epochs:
            print(f"    * Epoch {epoch:02d}/{epochs:02d} | Margin Hinge Loss: {total_loss/max(1, n_batches):.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
            
    # 5. Held-Out Subject Evaluation
    acc_5s, c_5s, n_5s, _ = evaluate_loso_fold(model, test_eeg_list, test_ya_list, test_yb_list, 5.0, fs, device)
    acc_10s, c_10s, n_10s, _ = evaluate_loso_fold(model, test_eeg_list, test_ya_list, test_yb_list, 10.0, fs, device)
    acc_20s, c_20s, n_20s, _ = evaluate_loso_fold(model, test_eeg_list, test_ya_list, test_yb_list, 20.0, fs, device)
    
    trial_metrics = evaluate_full_trial_decisions(model, test_eeg_list, test_ya_list, test_yb_list, 5.0, 0.5, fs, device)
    
    print("\n" + "-" * 92)
    print(f"  HELD-OUT LOSO RESULTS: Target Subject {test_sub}")
    print(f"    - 5.0s Window 2AFC:         {acc_5s:5.2f}% ({c_5s}/{n_5s} windows)")
    print(f"    - 10.0s Window 2AFC:        {acc_10s:5.2f}% ({c_10s}/{n_10s} windows)")
    print(f"    - 20.0s Window 2AFC:        {acc_20s:5.2f}% ({c_20s}/{n_20s} windows)")
    print(f"    - Trial Majority Win Acc:   {trial_metrics['majority_acc']:5.1f}% ({trial_metrics['majority_correct']}/{trial_metrics['total_trials']} trials)")
    print(f"    - Cumulative Margin Winner: {trial_metrics['cumulative_acc']:5.1f}% ({trial_metrics['cumulative_correct']}/{trial_metrics['total_trials']} trials)")
    print(f"    - Mean Time Locked on GT:   {trial_metrics['mean_lock_pct']:5.1f}%")
    print(f"    - Grand Mean Neural Margin: {trial_metrics['mean_margin']:+5.2f}")
    print("-" * 92)
    
    # Save fold checkpoint if requested
    if args.save_checkpoints:
        ckpt_dir = Path("/kaggle/working/loso_checkpoints") if Path("/kaggle/working").exists() else Path("checkpoints/loso")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_dir / f"catcn_loso_{test_sub}.pt")
        
    return {
        "subject": test_sub,
        "acc_5s": acc_5s,
        "acc_10s": acc_10s,
        "acc_20s": acc_20s,
        "majority_acc": trial_metrics["majority_acc"],
        "cumulative_acc": trial_metrics["cumulative_acc"],
        "mean_lock_pct": trial_metrics["mean_lock_pct"],
        "mean_margin": trial_metrics["mean_margin"],
        "total_trials": trial_metrics["total_trials"]
    }

def main():
    parser = argparse.ArgumentParser(description="CA-TCN Leave-One-Subject-Out (LOSO) Training & Evaluation")
    parser.add_argument("--subject", type=str, default="", help="Single held-out subject to evaluate (e.g. S1)")
    parser.add_argument("--all_folds", action="store_true", help="Run full 18-fold LOSO cross-validation across all subjects")
    parser.add_argument("--folds", type=str, default="", help="Comma-separated folds to evaluate (e.g. S1,S2,S7,S8,S15)")
    parser.add_argument("--montage", type=str, default="near_ear_expanded", help="Electrode montage")
    parser.add_argument("--window_sec", type=float, default=5.0, help="Window size in seconds")
    parser.add_argument("--hop_sec", type=float, default=2.5, help="Training hop size in seconds")
    parser.add_argument("--lowcut", type=float, default=1.0, help="EEG bandpass lowcut")
    parser.add_argument("--highcut", type=float, default=6.0, help="EEG bandpass highcut")
    parser.add_argument("--epochs", type=int, default=8, help="Training epochs per fold")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=64, help="CA-TCN hidden dimension")
    parser.add_argument("--smoke_test", action="store_true", help="Fast smoke test with few trials")
    parser.add_argument("--save_checkpoints", action="store_true", help="Save model weights per fold")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 96)
    print("  CA-TCN LEAVE-ONE-SUBJECT-OUT (LOSO) GENERALIZATION BENCHMARK")
    print(f"  Device: {device} | Montage: {args.montage} (8 channels) | Epochs per fold: {args.epochs}")
    print("=" * 96)
    
    montage_channels = MONTAGES[args.montage]
    all_paths = subject_files()
    
    if not all_paths:
        print("[ERROR] No DTU subject files found. Make sure dataset is mounted.")
        return
        
    mapping, envelopes = get_mapping_data("gammatone")
    
    # Determine target subjects
    if args.all_folds:
        target_subs = [f"S{i}" for i in range(1, 19)]
    elif args.folds:
        target_subs = [s.strip() for s in args.folds.split(",") if s.strip()]
    elif args.subject:
        target_subs = [args.subject]
    else:
        target_subs = ["S1"]
        
    all_loso_results = []
    
    t_start_total = time.time()
    for sub in target_subs:
        res = train_loso_fold(sub, all_paths, montage_channels, mapping, envelopes, args, device)
        if res is not None:
            all_loso_results.append(res)
            
    # Grand Comparison Report
    if all_loso_results:
        print("\n" + "=" * 110)
        print("  LOSO (ZERO-SHOT) vs. LOTO (PERSONALIZED) COMPARATIVE BENCHMARK TABLE")
        print("=" * 110)
        print(f"  {'Subject':<10} | {'5.0s 2AFC':<12} | {'10.0s 2AFC':<12} | {'20.0s 2AFC':<12} | {'Majority Win Acc':<18} | {'Mean Margin'}")
        print("  " + "-" * 106)
        
        for r in all_loso_results:
            print(f"  {r['subject']:<10} | {r['acc_5s']:>10.1f}% | {r['acc_10s']:>10.1f}% | {r['acc_20s']:>10.1f}% | {r['majority_acc']:>16.1f}% | {r['mean_margin']:+6.2f}")
            
        print("  " + "-" * 106)
        mean_5s = np.mean([r["acc_5s"] for r in all_loso_results])
        mean_10s = np.mean([r["acc_10s"] for r in all_loso_results])
        mean_20s = np.mean([r["acc_20s"] for r in all_loso_results])
        mean_maj = np.mean([r["majority_acc"] for r in all_loso_results])
        mean_mar = np.mean([r["mean_margin"] for r in all_loso_results])
        print(f"  {'LOSO GRAND MEAN':<10} | {mean_5s:>10.1f}% | {mean_10s:>10.1f}% | {mean_20s:>10.1f}% | {mean_maj:>16.1f}% | {mean_mar:+6.2f}")
        print("=" * 110)
        print(f"  Total Benchmark Execution Time: {(time.time() - t_start_total)/60.0:.1f} minutes.")
        print("=" * 110)

if __name__ == "__main__":
    main()
