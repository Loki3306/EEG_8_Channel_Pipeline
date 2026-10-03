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
from baselines.ridge_aad import load_subject_examples, subject_files, iter_leave_one_subject_out
from training.train_matchnet_wavlm import (
    FS, TRAIN_WINDOW_SEC,
    prepare_dataset, get_mapping_data
)
from training.montages import MONTAGES, DTU_CHANNELS

@torch.no_grad()
def evaluate_catcn_multiwindow_batched(model, X, Y_A, Y_B, device, batch_size=256):
    model.eval()
    samples_1s = int(1 * FS)
    samples_5s = int(5 * FS)
    
    trial_sub_1s = []
    trial_sub_5s = []
    
    for i in range(len(X)):
        x_np, ya_np, yb_np = X[i], Y_A[i], Y_B[i]
        trial_len = x_np.shape[1]
        
        # 1-second chunks
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

    windows_all = [1, 2, 5, 10, 20, 40]
    results = {}
    
    for w in windows_all:
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
        acc_accum_1s = c_accum_1s / max(n_accum_1s, 1)
        results[w] = acc_accum_1s
        
    return results

def run_ablation_test(args):
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
        print(f"Error: Target subject {target_sub} not found.")
        return

    print("=" * 88)
    print(" CA-TCN ANTI-CHEATING ABLATION SUITE (NEGATIVE CONTROLS)")
    print(f" Target Evaluation Subject: {target_sub}")
    print(f" Evaluation Montage: {args.montage} ({len(MONTAGES[args.montage])} channels)")
    print(f" Device: {device} | Training Epochs: {args.epochs}")
    print("=" * 88)

    # 1. Load Data
    mapping, envelopes = get_mapping_data("gammatone")
    montage_channels = MONTAGES[args.montage]

    print("\n[Stage 1]: Loading & Preprocessing Training Data...")
    X_tr_list, YA_tr_list, YB_tr_list = [], [], []
    X_va_list, YA_va_list, YB_va_list = [], [], []

    win_samples = int(TRAIN_WINDOW_SEC * FS)
    hop_samples = int(args.train_hop_sec * FS)

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
                X_tr_list.append(x[:, start:end])
                YA_tr_list.append(ya[:, start:end])
                YB_tr_list.append(yb[:, start:end])
                start += hop_samples

        for idx in perm[:val_split]:
            x, ya, yb = X_sub[idx], YA_sub[idx], YB_sub[idx]
            t_len = x.shape[1]
            start = 0
            while start + win_samples <= t_len:
                end = start + win_samples
                X_va_list.append(x[:, start:end])
                YA_va_list.append(ya[:, start:end])
                YB_va_list.append(yb[:, start:end])
                start += hop_samples

    X_tr_t = torch.from_numpy(np.stack(X_tr_list, axis=0))
    YA_tr_t = torch.from_numpy(np.stack(YA_tr_list, axis=0))
    YB_tr_t = torch.from_numpy(np.stack(YB_tr_list, axis=0))

    X_va_t = torch.from_numpy(np.stack(X_va_list, axis=0))
    YA_va_t = torch.from_numpy(np.stack(YA_va_list, axis=0))
    YB_va_t = torch.from_numpy(np.stack(YB_va_list, axis=0))

    print(f"  * Training Chunks: {X_tr_t.shape[0]} | Validation Chunks: {X_va_t.shape[0]}")

    train_ds = TensorDataset(X_tr_t, YA_tr_t, YB_tr_t)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=0)

    val_ds = TensorDataset(X_va_t, YA_va_t, YB_va_t)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=0)

    # 2. Train Standard Model on Training Fold
    print(f"\n[Stage 2]: Training CA-TCN on {len(train_paths)} training subjects...")
    model = CATCNDirectDecoder(
        eeg_channels=len(montage_channels),
        audio_channels=1,
        hidden_dim=64,
        max_lag_samples=8,
        dropout=0.2
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    best_val_loss = float('inf')
    best_weights = deepcopy(model.state_dict())

    for epoch in range(args.epochs):
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

    model.load_state_dict(best_weights)
    print("  -> Model Training Complete.")

    # 3. Load Test Data for Held-out Subject
    print(f"\n[Stage 3]: Loading Held-out Evaluation Subject: {target_sub}...")
    test_exs = list(load_subject_examples(held_out_path))
    X_te, YA_te, YB_te = prepare_dataset(
        test_exs, montage_channels, args.lowcut, args.highcut, target_sub, mapping, envelopes
    )
    YA_te = [ya.mean(axis=0, keepdims=True).astype(np.float32) if ya.shape[0] > 1 else ya.astype(np.float32) for ya in YA_te]
    YB_te = [yb.mean(axis=0, keepdims=True).astype(np.float32) if yb.shape[0] > 1 else yb.astype(np.float32) for yb in YB_te]

    # ==============================================================
    # RUN THE ABLATION EXPERIMENTS
    # ==============================================================
    ablation_results = {}

    # 1. BASELINE (Unaltered, Ground Truth)
    print("\n--- Running Control 0: Ground Truth Baseline ---")
    res_base = evaluate_catcn_multiwindow_batched(model, X_te, YA_te, YB_te, device)
    ablation_results["Ground Truth Baseline"] = res_base

    # 2. TIME-REVERSED AUDIO ENVELOPE (Ablation 1)
    # Reverses temporal envelope in time: preserves power spectrum & energy, destroys phase synchrony
    print("--- Running Control 1: Time-Reversed Speech Envelope ---")
    YA_rev = [np.ascontiguousarray(ya[:, ::-1]) for ya in YA_te]
    YB_rev = [np.ascontiguousarray(yb[:, ::-1]) for yb in YB_te]
    res_rev = evaluate_catcn_multiwindow_batched(model, X_te, YA_rev, YB_rev, device)
    ablation_results["Time-Reversed Audio"] = res_rev

    # 3. TEMPORAL LAG SHIFT (Ablation 2)
    # Circularly shifts audio by +10 seconds: breaks physiological latency window (0-250ms)
    print("--- Running Control 2: Temporal Latency Violation (+10s Lag Trap) ---")
    shift_samples = int(10.0 * FS)
    YA_shift = [np.roll(ya, shift_samples, axis=1) for ya in YA_te]
    YB_shift = [np.roll(yb, shift_samples, axis=1) for yb in YB_te]
    res_shift = evaluate_catcn_multiwindow_batched(model, X_te, YA_shift, YB_shift, device)
    ablation_results["Temporal Jitter (+10s Shift)"] = res_shift

    # 4. RANDOM LABEL PERMUTATION (Ablation 3)
    # Randomly flips YA and YB stream assignments
    print("--- Running Control 3: Random Label Permutation (Shuffled YA/YB) ---")
    rng = np.random.RandomState(42)
    YA_perm, YB_perm = [], []
    for ya, yb in zip(YA_te, YB_te):
        if rng.rand() > 0.5:
            YA_perm.append(yb)
            YB_perm.append(ya)
        else:
            YA_perm.append(ya)
            YB_perm.append(yb)
    res_perm = evaluate_catcn_multiwindow_batched(model, X_te, YA_perm, YB_perm, device)
    ablation_results["Label Permutation (Shuffled)"] = res_perm

    # 5. SYNTHETIC EEG GAUSSIAN NOISE (Ablation 4)
    # Replaces EEG with Gaussian noise matched to empirical mean and variance
    print("--- Running Control 4: Synthetic EEG Noise Control ---")
    X_noise = []
    for x in X_te:
        mean = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True)
        noise = rng.randn(*x.shape).astype(np.float32) * std + mean
        X_noise.append(noise)
    res_noise = evaluate_catcn_multiwindow_batched(model, X_noise, YA_te, YB_te, device)
    ablation_results["Synthetic EEG Noise"] = res_noise

    # PRINT FINAL RESULTS TABLE
    print("\n" + "=" * 92)
    print(" ANTI-CHEATING ABLATION SUMMARY TABLE")
    print(f" Subject: {target_sub} | Montage: {args.montage} (8 channels)")
    print("=" * 92)
    print(f"{'Condition':<36} {'Expected':<16} {'5s':>8} {'10s':>8} {'20s':>8} {'40s':>8}")
    print("-" * 92)

    expected_dict = {
        "Ground Truth Baseline": "High (>80%)",
        "Time-Reversed Audio": "Chance (~50%)",
        "Temporal Jitter (+10s Shift)": "Chance (~50%)",
        "Label Permutation (Shuffled)": "Chance (~50%)",
        "Synthetic EEG Noise": "Chance (~50%)"
    }

    out_csv = ["Condition,Expected,5s,10s,20s,40s"]

    for cond, res in ablation_results.items():
        exp = expected_dict[cond]
        a5 = res[5] * 100
        a10 = res[10] * 100
        a20 = res[20] * 100
        a40 = res[40] * 100
        print(f"{cond:<36} {exp:<16} {a5:>7.1f}% {a10:>7.1f}% {a20:>7.1f}% {a40:>7.1f}%")
        out_csv.append(f"{cond},{exp},{a5:.1f},{a10:.1f},{a20:.1f},{a40:.1f}")

    print("=" * 92)
    out_dir = Path(__file__).resolve().parents[3] / "results" / "ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"ablation_{target_sub}.csv", "w") as f:
        f.write("\n".join(out_csv))
    print(f"\nAblation artifact saved to: {out_dir / f'ablation_{target_sub}.csv'}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CA-TCN Anti-Cheating Ablation Suite")
    parser.add_argument("--subject", type=str, default="S1_data_preproc", help="Subject to evaluate")
    parser.add_argument("--montage", type=str, default="near_ear_expanded", help="Montage to evaluate")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--train_hop_sec", type=float, default=2.5)
    parser.add_argument("--lowcut", type=float, default=1.0)
    parser.add_argument("--highcut", type=float, default=6.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    run_ablation_test(args)
