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

REPO_ROOT = Path(__file__).resolve().parents[3]
VERIFY_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(VERIFY_ROOT) not in sys.path:
    sys.path.insert(0, str(VERIFY_ROOT))

from src.streaming.causal_filters import StreamingCausalEEGFilter
from models.catcn import CATCNDirectDecoder
from baselines.ridge_aad import load_subject_examples, subject_files, iter_leave_one_subject_out
from training.train_matchnet_wavlm import (
    FS, TRAIN_WINDOW_SEC,
    prepare_dataset, get_mapping_data
)
from training.montages import MONTAGES, DTU_CHANNELS

def butter_lowpass_sosfilt(data: np.ndarray, cutoff: float, fs: float, order: int = 2) -> np.ndarray:
    from scipy import signal
    sos = signal.butter(order, cutoff, btype='low', fs=fs, output='sos')
    return signal.sosfilt(sos, data).astype(np.float32)

@torch.no_grad()
def evaluate_catcn_windows(model, eeg_list, ya_list, yb_list, windows=[5, 10, 20, 40], fs=FS, device="cuda"):
    model.eval()
    results = {}
    for w in windows:
        w_samples = int(w * fs)
        deltas = []
        for eeg, ya, yb in zip(eeg_list, ya_list, yb_list):
            t_len = min(len(eeg), len(ya), len(yb))
            for s in range(0, t_len - w_samples + 1, w_samples):
                e = s + w_samples
                w_e = torch.from_numpy(eeg[s:e].T.copy()).unsqueeze(0).float().to(device)
                w_a = torch.from_numpy(ya[s:e].copy()).unsqueeze(0).unsqueeze(0).float().to(device)
                w_b = torch.from_numpy(yb[s:e].copy()).unsqueeze(0).unsqueeze(0).float().to(device)
                d, _, _ = model(w_e, w_a, w_b)
                deltas.append(d.item())
        deltas = np.array(deltas)
        if len(deltas) == 0:
            results[w] = 0.5
        else:
            acc = float(np.sum(deltas > 0)) / float(len(deltas))
            results[w] = acc
    return results

def run_ablation_test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_paths = subject_files()
    if not all_paths:
        print("No subjects found.")
        return

    target_sub = args.subject.split("_")[0]
    held_out_path = None
    train_paths = []
    for p in all_paths:
        if p.stem.split("_")[0] == target_sub:
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

    model = CATCNDirectDecoder(
        eeg_channels=len(montage_channels),
        audio_channels=1,
        hidden_dim=64,
        max_lag_samples=8,
        dropout=0.2
    ).to(device)

    ckpt_candidate = Path(args.checkpoint_path) if args.checkpoint_path else None
    if ckpt_candidate is None:
        for c in [
            Path(f"/kaggle/working/loso_checkpoints/catcn_adapted_{target_sub}.pt"),
            Path(f"/kaggle/working/loso_checkpoints/catcn_loso_{target_sub}.pt"),
            Path(f"checkpoints/loso/catcn_loso_{target_sub}.pt"),
            Path(f"checkpoints/catcn_loso_{target_sub}.pt")
        ]:
            if c.exists():
                ckpt_candidate = c
                break

    if ckpt_candidate and ckpt_candidate.exists():
        print(f"\n[MODEL] Loading pre-trained checkpoint from: {ckpt_candidate} (skipping Stage 1 & 2 training)")
        try:
            ckpt = torch.load(str(ckpt_candidate), map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(str(ckpt_candidate), map_location=device)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        elif "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)
    else:
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
    print(f"
[Stage 3]: Loading Held-out Evaluation Subject: {target_sub}...")
    causal_filter = StreamingCausalEEGFilter(fs=FS, lowcut=1.0, highcut=6.0, order=2, n_channels=len(montage_channels))
    test_exs = list(load_subject_examples(held_out_path))
    _, ya_raw, yb_raw = prepare_dataset(test_exs, montage_channels, 1.0, 6.0, target_sub, mapping, envelopes)
    ya_clean = [ya.mean(axis=0, keepdims=True).squeeze(0).astype(np.float32) if ya.shape[0] > 1 else ya.squeeze(0).astype(np.float32) for ya in ya_raw]
    yb_clean = [yb.mean(axis=0, keepdims=True).squeeze(0).astype(np.float32) if yb.shape[0] > 1 else yb.squeeze(0).astype(np.float32) for yb in yb_raw]

    eeg_all, ya_all, yb_all = [], [], []
    for idx in range(min(len(test_exs), len(ya_clean))):
        raw_eeg = test_exs[idx].eeg[:, montage_channels].astype(np.float32)
        min_len = min(len(raw_eeg), len(ya_clean[idx]), len(yb_clean[idx]))
        raw_eeg = raw_eeg[:min_len]

        causal_filter.reset()
        eeg_c = causal_filter.process_chunk(raw_eeg)
        eeg_c = (eeg_c - np.mean(eeg_c, axis=0, keepdims=True)) / (np.std(eeg_c, axis=0, keepdims=True) + 1e-8)

        ya_c = butter_lowpass_sosfilt(ya_clean[idx][:min_len], 8.0, FS, order=2).astype(np.float32)
        yb_c = butter_lowpass_sosfilt(yb_clean[idx][:min_len], 8.0, FS, order=2).astype(np.float32)
        ya_c = (ya_c - np.mean(ya_c)) / (np.std(ya_c) + 1e-8)
        yb_c = (yb_c - np.mean(yb_c)) / (np.std(yb_c) + 1e-8)

        eeg_all.append(eeg_c)
        ya_all.append(ya_c)
        yb_all.append(yb_c)

    # ==============================================================
    # RUN THE ABLATION EXPERIMENTS
    # ==============================================================
    ablation_results = {}

    # 1. BASELINE (Unaltered, Ground Truth)
    print("
--- Running Control 0: Ground Truth Baseline ---")
    res_base = evaluate_catcn_windows(model, eeg_all, ya_all, yb_all, windows=[5, 10, 20, 40], fs=FS, device=device)
    ablation_results["Ground Truth Baseline"] = res_base

    # 2. TIME-REVERSED AUDIO ENVELOPE (Ablation 1)
    print("--- Running Control 1: Time-Reversed Speech Envelope ---")
    ya_rev = [np.ascontiguousarray(ya[::-1]) for ya in ya_all]
    yb_rev = [np.ascontiguousarray(yb[::-1]) for yb in yb_all]
    res_rev = evaluate_catcn_windows(model, eeg_all, ya_rev, yb_rev, windows=[5, 10, 20, 40], fs=FS, device=device)
    ablation_results["Time-Reversed Audio"] = res_rev

    # 3. TEMPORAL LAG SHIFT (Ablation 2)
    print("--- Running Control 2: Temporal Latency Violation (+10s Lag Trap) ---")
    shift_samples = int(10.0 * FS)
    ya_shift = [np.roll(ya, shift_samples) for ya in ya_all]
    yb_shift = [np.roll(yb, shift_samples) for yb in yb_all]
    res_shift = evaluate_catcn_windows(model, eeg_all, ya_shift, yb_shift, windows=[5, 10, 20, 40], fs=FS, device=device)
    ablation_results["Temporal Jitter (+10s Shift)"] = res_shift

    # 4. RANDOM LABEL PERMUTATION (Ablation 3)
    print("--- Running Control 3: Random Label Permutation (Shuffled YA/YB) ---")
    rng = np.random.RandomState(42)
    ya_perm, yb_perm = [], []
    for ya, yb in zip(ya_all, yb_all):
        if rng.rand() > 0.5:
            ya_perm.append(yb)
            yb_perm.append(ya)
        else:
            ya_perm.append(ya)
            yb_perm.append(yb)
    res_perm = evaluate_catcn_windows(model, eeg_all, ya_perm, yb_perm, windows=[5, 10, 20, 40], fs=FS, device=device)
    ablation_results["Label Permutation (Shuffled)"] = res_perm

    # 5. SYNTHETIC EEG GAUSSIAN NOISE (Ablation 4)
    print("--- Running Control 4: Synthetic EEG Noise Control ---")
    eeg_noise = []
    for e in eeg_all:
        mean = e.mean(axis=0, keepdims=True)
        std = e.std(axis=0, keepdims=True)
        noise = rng.randn(*e.shape).astype(np.float32) * std + mean
        eeg_noise.append(noise)
    res_noise = evaluate_catcn_windows(model, eeg_noise, ya_all, yb_all, windows=[5, 10, 20, 40], fs=FS, device=device)
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
    out_dir = Path("/kaggle/working/results/ablations") if Path("/kaggle/working").exists() else (Path(__file__).resolve().parents[3] / "results" / "ablations")
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
    parser.add_argument("--checkpoint_path", type=str, default="", help="Path to pre-trained checkpoint to evaluate directly without retraining")
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    run_ablation_test(args)
