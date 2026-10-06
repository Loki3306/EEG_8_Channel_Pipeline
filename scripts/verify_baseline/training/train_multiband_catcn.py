"""
Universal Multi-Band Cochlear Gammatone + Causal ERP Cross-Attention CA-TCN Training Pipeline.
Supports training across all 18 DTU subjects with 8-subband tonotopic cochlear inputs
and causal ERP cross-attention tracking.

Designed for seamless execution on Kaggle GPU and rapid local CPU smoke verification.
"""

from __future__ import annotations
import argparse
import sys
import os
import json
import time
import math
import glob
from pathlib import Path
from copy import deepcopy
import numpy as np
from scipy import signal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[3]
VERIFY_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(VERIFY_ROOT) not in sys.path:
    sys.path.insert(0, str(VERIFY_ROOT))

from models.multiband_catcn import MultiBandCATCNDecoder
from src.streaming.causal_filters import StreamingCausalEEGFilter
from baselines.ridge_aad import load_subject_examples, TrialExample
from training.montages import MONTAGES, DTU_CHANNELS
from data.multiband_provider import get_multiband_envelopes, get_mapping

FS = 64

# Prior single-band 5.0s CA-TCN benchmarks across all 18 DTU subjects
BASELINE_5S = {
    "S1": 66.3, "S2": 67.4, "S3": 60.2, "S4": 64.5, "S5": 65.1,
    "S6": 54.4, "S7": 76.0, "S8": 72.6, "S9": 61.1, "S10": 63.2,
    "S11": 57.3, "S12": 66.8, "S13": 69.8, "S14": 66.5, "S15": 78.5,
    "S16": 60.7, "S17": 64.6, "S18": 66.7
}

def resolve_output_path(path_str: str) -> Path:
    p = Path(path_str)
    is_kaggle = (os.name != 'nt') and Path("/kaggle").exists()
    if not is_kaggle and "kaggle" in str(p).lower():
        local_dir = REPO_ROOT / "results" / "multiband_catcn"
        local_dir.mkdir(parents=True, exist_ok=True)
        return local_dir / p.name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def butter_lowpass_sosfilt(data: np.ndarray, cutoff: float, fs: float, order: int = 2) -> np.ndarray:
    """Causal low-pass filter for multi-band audio envelopes along the time axis."""
    sos = signal.butter(order, cutoff, btype='low', fs=fs, output='sos')
    if data.ndim == 1:
        zi = signal.sosfilt_zi(sos) * (data[0] if len(data) > 0 else 0.0)
        out, _ = signal.sosfilt(sos, data, zi=zi)
        return out
    else:
        # Multi-band: shape [Bands, Time]
        out = np.zeros_like(data)
        for b in range(data.shape[0]):
            zi = signal.sosfilt_zi(sos) * (data[b, 0] if data.shape[1] > 0 else 0.0)
            out[b], _ = signal.sosfilt(sos, data[b], zi=zi)
        return out

def evaluate_windows(model, eeg_list, ya_list, yb_list, label_list, window_samples, device):
    """Evaluates non-overlapping windows across held-out test trials against ground truth labels."""
    total_wins = 0
    correct_wins = 0
    model.eval()
    with torch.no_grad():
        for eeg, ya, yb, trial_label in zip(eeg_list, ya_list, yb_list, label_list):
            t_len = min(len(eeg), ya.shape[-1], yb.shape[-1])
            for s in range(0, t_len - window_samples + 1, window_samples):
                e = s + window_samples
                w_e = torch.from_numpy(eeg[s:e].T.copy()).unsqueeze(0).float().to(device)
                w_a = torch.from_numpy(ya[:, s:e].copy()).unsqueeze(0).float().to(device)
                w_b = torch.from_numpy(yb[:, s:e].copy()).unsqueeze(0).float().to(device)
                delta, (la, lb), _ = model(w_e, w_a, w_b)
                # Model decision: Choose Stream A if delta > 0, Stream B if delta < 0
                pred_label = 1 if delta.item() > 0 else 2
                if pred_label == trial_label:
                    correct_wins += 1
                total_wins += 1
                
    if total_wins == 0:
        return 50.0, 0
    acc = (correct_wins / total_wins) * 100.0
    return acc, total_wins

def discover_eeg_subjects(custom_eeg_dir: str = None) -> list[Path]:
    """Discovers all DTU S*_data_preproc.mat files with exhaustive error-tolerant search."""
    found_paths = set()
    
    # Priority 1: If custom_eeg_dir provided directly
    if custom_eeg_dir:
        p = Path(custom_eeg_dir)
        if p.is_file():
            found_paths.add(p)
        elif p.is_dir():
            try:
                for f in p.iterdir():
                    if f.suffix.lower() == ".mat":
                        found_paths.add(f)
            except Exception:
                pass
            try:
                for f in p.rglob("*.mat"):
                    found_paths.add(f)
            except Exception:
                pass
                
    # Priority 2: Standard Kaggle and local candidate paths
    standard_candidates = [
        Path("/kaggle/input/datasets/lokeshgile/dataset-eeg"),
        Path("/kaggle/input/dataset-eeg"),
        Path("/kaggle/input/Dataset_EEG"),
        Path("/kaggle/input/dataset_eeg"),
        Path("/kaggle/input/datasets/lokeshgile/dtu-eeg-raw"),
        Path("/kaggle/input/dtu-eeg-raw"),
        Path(r"C:\Users\lokes\Downloads\archive (2)\DATA_preproc"),
        REPO_ROOT / "data",
        VERIFY_ROOT / "data",
    ]
    for cand in standard_candidates:
        if cand.exists() and cand.is_dir():
            try:
                for f in cand.iterdir():
                    if f.suffix.lower() == ".mat":
                        found_paths.add(f)
            except Exception:
                pass
            try:
                for f in cand.rglob("*.mat"):
                    found_paths.add(f)
            except Exception:
                pass

    # Priority 3: Deep exhaustive search across /kaggle/input using both glob and os.walk with error suppression
    if Path("/kaggle/input").exists():
        try:
            for m in glob.glob("/kaggle/input/**/*.mat", recursive=True):
                found_paths.add(Path(m))
        except Exception:
            pass
            
        def ignore_err(e):
            return None
            
        for root, _, files in os.walk("/kaggle/input", onerror=ignore_err, followlinks=True):
            for f in files:
                if f.lower().endswith(".mat"):
                    found_paths.add(Path(root) / f)

    # Filter to actual DTU subject .mat files (contain data_preproc or start with S[0-9]+)
    valid_subjects = []
    for p in found_paths:
        name = p.name.lower()
        if "data_preproc" in name or (name.startswith("s") and len(name) > 1 and name[1:2].isdigit()):
            valid_subjects.append(p)
            
    if not valid_subjects:
        return []
        
    def sort_key(p: Path):
        s_part = p.stem.split("_")[0].upper()
        if s_part.startswith("S") and s_part[1:].isdigit():
            return (0, int(s_part[1:]))
        return (1, p.stem)
        
    # Deduplicate by filename stem so identical files from symlinks/aliases don't duplicate
    by_stem = {p.stem: p for p in valid_subjects}
    return sorted(by_stem.values(), key=sort_key)

def run_multiband_training(args):
    montage_channels = MONTAGES[args.montage]
    n_ch = len(montage_channels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.smoke_test:
        args.epochs = 1
        args.batch_size = min(args.batch_size, 16)
        print("\n" + "=" * 96)
        print("  [SMOKE TEST MODE ENABLED]: Running rapid 1-epoch pipeline verification on CPU")
        print("=" * 96)
        
    print("=" * 96)
    print("  MULTIBAND COCHLEAR GAMMATONE + CAUSAL ERP CROSS-ATTENTION CA-TCN TRAINING")
    print(f"  Montage: {args.montage} ({n_ch} channels) | Audio Subbands: {args.audio_bands}")
    print(f"  Device: {device} | Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print("=" * 96)
    
    # 1. Discover Subjects
    all_paths = discover_eeg_subjects(args.eeg_dir)
    print(f"[DATA] Discovered {len(all_paths)} total subject file(s) on disk: {[p.name for p in all_paths]}")
    
    if args.subjects and args.subjects.lower() != 'all':
        requested = [s.strip().upper() for s in args.subjects.split(",") if s.strip()]
        filtered = [p for p in all_paths if p.stem.split("_")[0].upper() in requested]
        if len(filtered) > 0:
            all_paths = filtered
            print(f"[DATA] Filtered to {len(all_paths)} requested subject(s): {[p.stem for p in all_paths]}")
        else:
            print(f"[WARNING] None of the requested subjects {requested} were found in the available files on disk.")
            print(f"[INFO] Available subject files: {[p.name for p in all_paths]}")
            print(f"[INFO] Automatically proceeding with all {len(all_paths)} discovered subject file(s).")
            
    if not all_paths:
        if args.smoke_test:
            print("[SMOKE TEST] No real DTU files on disk. Synthesizing 2 mock subjects for pipeline audit...")
            all_paths = [Path("S1_data_preproc.mat"), Path("S2_data_preproc.mat")]
        else:
            raise FileNotFoundError(
                f"No DTU patient files found matching criteria. Checked custom eeg_dir: '{args.eeg_dir}' and /kaggle/input."
            )
    elif args.smoke_test and not args.subjects:
        all_paths = all_paths[:2]
        
    print(f"[DATA] Selected {len(all_paths)} DTU subjects for training: {[p.stem for p in all_paths]}")
    
    # 2. Load Mapping and 8-band Envelopes
    mapping = {}
    envelopes = {}
    try:
        mapping = get_mapping()
        envelopes = get_multiband_envelopes(
            target_bands=args.audio_bands,
            custom_env_file=args.audio_env_file,
            custom_audio_dir=args.audio_dir,
            auto_extract=(not args.smoke_test)
        )
    except Exception as e:
        if args.smoke_test:
            print(f"[SMOKE TEST] Audio data not available ({e}). Synthesizing {args.audio_bands}-band envelopes.")
        else:
            raise e

    win_samples = int(args.window_sec * FS)
    causal_eeg_filter = StreamingCausalEEGFilter(lowcut=1.0, highcut=6.0, fs=FS, order=2, n_channels=n_ch)
    
    # 3. Process Subjects into Training/Validation Tensors
    print("\n[DATA PREPARATION]: Extracting causal streaming EEG and multi-band envelopes...")
    t_data_start = time.time()
    
    X_tr_list, S1_tr_list, S2_tr_list, TGT_tr_list = [], [], [], []
    X_va_list, S1_va_list, S2_va_list, TGT_va_list = [], [], [], []
    subject_test_data = {}
    total_train_trials = 0
    total_test_trials = 0
    
    for p in all_paths:
        sub_name = p.stem
        sub_key = sub_name.replace("_data_preproc", "")
        
        if args.smoke_test and (not p.exists() or not envelopes):
            # Synthesize 4 trials of DTU length (3200 samples = 50.0s @ 64 Hz)
            n_trials = 4
            exs = [
                TrialExample(
                    subject=sub_name,
                    trial_index=i,
                    eeg=np.random.randn(3200, 64).astype(np.float32),
                    wav_a=np.random.randn(3200).astype(np.float32),
                    wav_b=np.random.randn(3200).astype(np.float32),
                    label=(1 if i % 2 == 0 else 2)
                )
                for i in range(n_trials)
            ]
            raw_ya_list = [np.random.randn(args.audio_bands, 3200).astype(np.float32) for _ in range(n_trials)]
            raw_yb_list = [np.random.randn(args.audio_bands, 3200).astype(np.float32) for _ in range(n_trials)]
        else:
            exs = list(load_subject_examples(p))
            if args.smoke_test:
                exs = exs[:4]
                
            raw_ya_list = []
            raw_yb_list = []
            valid_exs = []
            for i, ex in enumerate(exs):
                trial_idx = getattr(ex, 'trial_index', i)
                trial_key = f"trial_{trial_idx}"
                if sub_key in mapping and trial_key in mapping[sub_key]:
                    fa = mapping[sub_key][trial_key]["wavA"]["filename"]
                    fb = mapping[sub_key][trial_key]["wavB"]["filename"]
                    if fa in envelopes and fb in envelopes:
                        raw_ya_list.append(envelopes[fa])
                        raw_yb_list.append(envelopes[fb])
                        valid_exs.append(ex)
            exs = valid_exs
            
        n_valid = min(len(exs), len(raw_ya_list))
        if n_valid == 0:
            continue
            
        split_idx = int(math.floor(n_valid * (1.0 - args.test_split)))
        sub_eeg_te, sub_ya_te, sub_yb_te, sub_lbl_te = [], [], [], []
        
        for idx in range(n_valid):
            trial_label = getattr(exs[idx], 'label', 1)  # 1 = Attend A, 2 = Attend B
            raw_eeg = exs[idx].eeg[:, montage_channels].astype(np.float32)
            cur_ya = raw_ya_list[idx]
            cur_yb = raw_yb_list[idx]
            min_len = min(len(raw_eeg), cur_ya.shape[-1], cur_yb.shape[-1])
            raw_eeg = raw_eeg[:min_len]
            cur_ya = cur_ya[:, :min_len]
            cur_yb = cur_yb[:, :min_len]
            
            # Causal EEG filtering + standardization
            causal_eeg_filter.reset()
            eeg_c = causal_eeg_filter.process_chunk(raw_eeg)
            eeg_c = (eeg_c - np.mean(eeg_c, axis=0, keepdims=True)) / (np.std(eeg_c, axis=0, keepdims=True) + 1e-12)
            
            # Causal Audio lowpass + standardization
            ya_c = butter_lowpass_sosfilt(cur_ya, 8.0, FS, order=2).astype(np.float32)
            yb_c = butter_lowpass_sosfilt(cur_yb, 8.0, FS, order=2).astype(np.float32)
            ya_c = (ya_c - np.mean(ya_c, axis=-1, keepdims=True)) / (np.std(ya_c, axis=-1, keepdims=True) + 1e-12)
            yb_c = (yb_c - np.mean(yb_c, axis=-1, keepdims=True)) / (np.std(yb_c, axis=-1, keepdims=True) + 1e-12)
            
            # Ground-truth attended vs unattended assignment:
            if trial_label == 1:
                cur_att, cur_unatt = ya_c, yb_c
            else:
                cur_att, cur_unatt = yb_c, ya_c
            
            if idx < split_idx:
                total_train_trials += 1
                x_t = eeg_c.T # [C_eeg, T]
                is_val_trial = (idx % 10 == 0)
                
                # Chunk training trials
                hop_samples = int(args.hop_sec * FS)
                start = 0
                w_count = 0
                while start + win_samples <= min_len:
                    end = start + win_samples
                    w_x = x_t[:, start:end]
                    # Symmetrized training to eliminate slot position bias:
                    # Alternate stream positions with ground-truth target:
                    # Target = +1.0 when Slot 1 is attended, -1.0 when Slot 2 is attended
                    if w_count % 2 == 0:
                        w_s1, w_s2, tgt = cur_att[:, start:end], cur_unatt[:, start:end], 1.0
                    else:
                        w_s1, w_s2, tgt = cur_unatt[:, start:end], cur_att[:, start:end], -1.0
                    
                    if is_val_trial:
                        X_va_list.append(w_x)
                        S1_va_list.append(w_s1)
                        S2_va_list.append(w_s2)
                        TGT_va_list.append(tgt)
                    else:
                        X_tr_list.append(w_x)
                        S1_tr_list.append(w_s1)
                        S2_tr_list.append(w_s2)
                        TGT_tr_list.append(tgt)
                    start += hop_samples
                    w_count += 1
            else:
                total_test_trials += 1
                sub_eeg_te.append(eeg_c)
                sub_ya_te.append(ya_c)
                sub_yb_te.append(yb_c)
                sub_lbl_te.append(trial_label)
                
        if sub_eeg_te:
            subject_test_data[sub_name] = (sub_eeg_te, sub_ya_te, sub_yb_te, sub_lbl_te)
            
    print(f"[DATA READY]: Extracted {len(X_tr_list)} train windows, {len(X_va_list)} val windows across {total_train_trials} trials in {time.time()-t_data_start:.1f}s.")
    
    if len(X_tr_list) == 0:
        raise RuntimeError("No training windows extracted. Please check dataset paths and envelopes.")
        
    X_tr = torch.from_numpy(np.stack(X_tr_list, axis=0)).float()
    S1_tr = torch.from_numpy(np.stack(S1_tr_list, axis=0)).float()
    S2_tr = torch.from_numpy(np.stack(S2_tr_list, axis=0)).float()
    TGT_tr = torch.from_numpy(np.array(TGT_tr_list, dtype=np.float32)).float()
    
    # Free memory
    del X_tr_list, S1_tr_list, S2_tr_list, TGT_tr_list
    import gc
    gc.collect()
    
    if len(X_va_list) > 0:
        X_va = torch.from_numpy(np.stack(X_va_list, axis=0)).float()
        S1_va = torch.from_numpy(np.stack(S1_va_list, axis=0)).float()
        S2_va = torch.from_numpy(np.stack(S2_va_list, axis=0)).float()
        TGT_va = torch.from_numpy(np.array(TGT_va_list, dtype=np.float32)).float()
        del X_va_list, S1_va_list, S2_va_list, TGT_va_list
        gc.collect()
    else:
        X_va = X_tr[:min(16, len(X_tr))]
        S1_va = S1_tr[:min(16, len(S1_tr))]
        S2_va = S2_tr[:min(16, len(S2_tr))]
        TGT_va = TGT_tr[:min(16, len(TGT_tr))]
        
    # GPU OPTIMIZATION: Pinned memory for async DMA transfers
    use_cuda = torch.cuda.is_available()
    train_loader = DataLoader(
        TensorDataset(X_tr, S1_tr, S2_tr, TGT_tr),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=(len(X_tr) > args.batch_size),
        pin_memory=use_cuda,
        num_workers=2 if use_cuda else 0
    )
    val_loader = DataLoader(
        TensorDataset(X_va, S1_va, S2_va, TGT_va),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=use_cuda,
        num_workers=2 if use_cuda else 0
    )
    
    # 4. Instantiate Multi-Band CA-TCN Model
    model = MultiBandCATCNDecoder(
        eeg_channels=n_ch,
        audio_bands=args.audio_bands,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        max_erp_samples=args.max_erp_samples,
        max_lag_samples=args.max_lag_samples,
        dropout=args.dropout
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[MODEL INITIALIZED]: MultiBand-CATCN with {n_params:,} trainable parameters.")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    
    best_val_loss = float('inf')
    best_weights = deepcopy(model.state_dict())
    
    # 5. Training Loop
    print("\n" + "=" * 96)
    print(f"  COMMENCING MULTIBAND TRAINING ({args.epochs} EPOCHS)")
    print("=" * 96)
    
    for epoch in range(1, args.epochs + 1):
        t_epoch_start = time.time()
        model.train()
        train_loss = 0.0
        n_train_batches = 0
        
        for bx, bs1, bs2, btgt in train_loader:
            bx = bx.to(device, non_blocking=True)
            bs1 = bs1.to(device, non_blocking=True)
            bs2 = bs2.to(device, non_blocking=True)
            btgt = btgt.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                delta, (l1, l2), _ = model(bx, bs1, bs2)
                # Symmetrized Margin Ranking Loss: btgt * delta > 0.5
                loss = torch.clamp(0.5 - btgt * delta, min=0.0).mean()
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            n_train_batches += 1
            
        scheduler.step()
        avg_train_loss = train_loss / max(1, n_train_batches)
        
        # Validation
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for bx, bs1, bs2, btgt in val_loader:
                bx = bx.to(device, non_blocking=True)
                bs1 = bs1.to(device, non_blocking=True)
                bs2 = bs2.to(device, non_blocking=True)
                btgt = btgt.to(device, non_blocking=True)
                with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                    delta, (l1, l2), _ = model(bx, bs1, bs2)
                    v_loss = torch.clamp(0.5 - btgt * delta, min=0.0).mean()
                val_loss += v_loss.item()
                n_val_batches += 1
                
        avg_val_loss = val_loss / max(1, n_val_batches)
        epoch_sec = time.time() - t_epoch_start
        
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            best_weights = deepcopy(model.state_dict())
            star_flag = " [*BEST*]"
        else:
            star_flag = ""
            
        print(f"  Epoch [{epoch:02d}/{args.epochs:02d}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | Time: {epoch_sec:.1f}s{star_flag}")
        
        if epoch % 5 == 0 or is_best or epoch == args.epochs:
            ckpt_path = resolve_output_path(args.output_model)
            torch.save(best_weights, ckpt_path)
            
    # 6. Final Evaluation on Held-Out Test Split
    print("\n" + "=" * 96)
    print("  GRAND COHORT EVALUATION ON HELD-OUT TRIALS (5.0s NON-OVERLAPPING WINDOWS)")
    print("=" * 96)
    model.load_state_dict(best_weights)
    model.eval()
    
    subject_results = {}
    cohort_accs = []
    
    print(f"  {'Subject':<16} | {'2AFC Accuracy':<14} | {'Baseline (5s)':<14} | {'Gain / Margin':<14} | {'Test Wins':<10}")
    print("  " + "-" * 76)
    for sub_name, (te_eeg, te_ya, te_yb, te_lbls) in subject_test_data.items():
        acc, n_wins = evaluate_windows(model, te_eeg, te_ya, te_yb, te_lbls, win_samples, device)
        s_key = sub_name.split("_")[0].upper()
        base_acc = BASELINE_5S.get(s_key, 65.7)
        gain = acc - base_acc
        gain_str = f"+{gain:.2f}%" if gain >= 0 else f"{gain:.2f}%"
        subject_results[sub_name] = {
            "accuracy": round(acc, 2),
            "baseline_5s": round(base_acc, 2),
            "gain": round(gain, 2),
            "test_windows": n_wins
        }
        cohort_accs.append(acc)
        print(f"  {sub_name:<16} | {acc:5.1f}%          | {base_acc:5.1f}%         | {gain_str:<14} | {n_wins:<10}")
        
    grand_mean = float(np.mean(cohort_accs)) if cohort_accs else 0.0
    print("-" * 96)
    print(f"  COHORT EVALUATION MEAN ACCURACY: {grand_mean:.2f}% across {len(subject_results)} subjects")
    print("=" * 96)
    
    # Save Metrics JSON
    metrics = {
        "architecture": "MultiBand-CATCN",
        "montage": args.montage,
        "n_channels": n_ch,
        "audio_bands": args.audio_bands,
        "epochs": args.epochs,
        "parameters": n_params,
        "best_val_loss": round(best_val_loss, 4),
        "grand_mean_accuracy": round(grand_mean, 2),
        "subject_accuracies": subject_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    metrics_path = resolve_output_path(args.output_metrics)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[OUTPUT] Model checkpoint saved to: {resolve_output_path(args.output_model)}")
    print(f"[OUTPUT] Metrics saved to: {metrics_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Band Cochlear Gammatone + Causal ERP Cross-Attention CA-TCN Training")
    parser.add_argument("--montage", type=str, default="dtu_8ch", choices=list(MONTAGES.keys()))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--window_sec", type=float, default=5.0)
    parser.add_argument("--hop_sec", type=float, default=1.0)
    parser.add_argument("--test_split", type=float, default=0.2)
    parser.add_argument("--audio_bands", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--max_erp_samples", type=int, default=22)
    parser.add_argument("--max_lag_samples", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--eeg_dir", type=str, default=None)
    parser.add_argument("--audio_dir", type=str, default=None)
    parser.add_argument("--audio_env_file", type=str, default=None)
    parser.add_argument("--output_model", type=str, default="/kaggle/working/multiband_catcn_best.pt")
    parser.add_argument("--output_metrics", type=str, default="/kaggle/working/multiband_catcn_metrics.json")
    parser.add_argument("--smoke_test", action="store_true", help="Run rapid CPU smoke test")
    parser.add_argument("--subjects", type=str, default=None, help="Comma-separated subjects to run, e.g. S15,S6,S1,S14")
    args = parser.parse_args()
    run_multiband_training(args)
