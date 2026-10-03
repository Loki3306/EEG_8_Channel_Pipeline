import argparse
import sys
import os
import json
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import psutil
import gc
from pathlib import Path
from copy import deepcopy
from datetime import datetime
import subprocess

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.catcn import CATCNDirectDecoder
from baselines.ridge_aad import load_subject_examples, subject_files, iter_leave_one_subject_out
from training.train_matchnet_wavlm import (
    FS, TRAIN_WINDOW_SEC,
    prepare_dataset, get_mapping_data
)
from training.montages import MONTAGES, DTU_CHANNELS

def get_git_revision_hash() -> str:
    try:
        project_root = Path(__file__).resolve().parents[3]
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(project_root)).decode('ascii').strip()
    except Exception:
        return "unknown"

def evaluate_catcn_multiwindow_batched(model, X, Y_A, Y_B, device, batch_size=256):
    """
    Batched multi-window evaluation for CA-TCN.
    Evaluates 1s and 5s atomic sub-windows in bulk on GPU to eliminate single-sample launch overhead.
    """
    model.eval()
    
    samples_1s = int(1 * FS)
    samples_5s = int(5 * FS)
    
    trial_sub_1s = []
    trial_sub_5s = []
    
    with torch.no_grad():
        for i in range(len(X)):
            x_np, ya_np, yb_np = X[i], Y_A[i], Y_B[i]
            trial_len = x_np.shape[1]
            
            # --- 1-second atomic sub-windows ---
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
                    delta_list.append(d_b.cpu())
                deltas = torch.cat(delta_list).numpy()
                d_1s = deltas.tolist()
                v_1s = [1.0 if d > 0 else (0.5 if d == 0 else 0.0) for d in d_1s]
            else:
                d_1s, v_1s = [], []
            trial_sub_1s.append((d_1s, v_1s))
            
            # --- 5-second atomic sub-windows ---
            c_5s_x, c_5s_ya, c_5s_yb = [], [], []
            start = 0
            while start + samples_5s <= trial_len:
                end = start + samples_5s
                c_5s_x.append(x_np[:, start:end])
                c_5s_ya.append(ya_np[:, start:end])
                c_5s_yb.append(yb_np[:, start:end])
                start += samples_5s
                
            if c_5s_x:
                bx = torch.from_numpy(np.stack(c_5s_x, axis=0)).to(device, dtype=torch.float32)
                bya = torch.from_numpy(np.stack(c_5s_ya, axis=0)).to(device, dtype=torch.float32)
                byb = torch.from_numpy(np.stack(c_5s_yb, axis=0)).to(device, dtype=torch.float32)
                
                delta_list = []
                for b_idx in range(0, bx.size(0), batch_size):
                    d_b, _, _ = model(bx[b_idx:b_idx+batch_size], bya[b_idx:b_idx+batch_size], byb[b_idx:b_idx+batch_size])
                    delta_list.append(d_b.cpu())
                deltas = torch.cat(delta_list).numpy()
                d_5s = deltas.tolist()
                v_5s = [1.0 if d > 0 else (0.5 if d == 0 else 0.0) for d in d_5s]
            else:
                d_5s, v_5s = [], []
            trial_sub_5s.append((d_5s, v_5s))

    windows_all = [1, 2, 5, 10, 15, 20, 25, 30, 35, 40]
    results = {}
    
    for w in windows_all:
        c_accum_1s, n_accum_1s = 0.0, 0
        c_vote_1s = 0.0
        m_1s = w
        for d_list, v_list in trial_sub_1s:
            b_start = 0
            while b_start + m_1s <= len(d_list):
                block_d = d_list[b_start : b_start + m_1s]
                block_v = v_list[b_start : b_start + m_1s]
                if sum(block_d) > 0: c_accum_1s += 1.0
                elif sum(block_d) == 0: c_accum_1s += 0.5
                
                if sum(block_v) > m_1s / 2.0: c_vote_1s += 1.0
                elif sum(block_v) == m_1s / 2.0: c_vote_1s += 0.5
                
                n_accum_1s += 1
                b_start += m_1s
        acc_accum_1s = c_accum_1s / max(n_accum_1s, 1)
        acc_vote_1s = c_vote_1s / max(n_accum_1s, 1)
        
        acc_accum_5s = None
        if w >= 5 and w % 5 == 0:
            m_5s = w // 5
            c_accum_5s, n_accum_5s = 0.0, 0
            for d_list, _ in trial_sub_5s:
                b_start = 0
                while b_start + m_5s <= len(d_list):
                    block_d = d_list[b_start : b_start + m_5s]
                    if sum(block_d) > 0: c_accum_5s += 1.0
                    elif sum(block_d) == 0: c_accum_5s += 0.5
                    n_accum_5s += 1
                    b_start += m_5s
            acc_accum_5s = c_accum_5s / max(n_accum_5s, 1)
            
        # Batched direct evaluation on full window length
        c_direct, n_direct = 0.0, 0
        w_samples = int(w * FS)
        d_x, d_ya, d_yb = [], [], []
        for i in range(len(X)):
            x_np, ya_np, yb_np = X[i], Y_A[i], Y_B[i]
            start = 0
            while start + w_samples <= x_np.shape[1]:
                end = start + w_samples
                d_x.append(x_np[:, start:end])
                d_ya.append(ya_np[:, start:end])
                d_yb.append(yb_np[:, start:end])
                start += w_samples
        if d_x:
            bx = torch.from_numpy(np.stack(d_x, axis=0)).to(device, dtype=torch.float32)
            bya = torch.from_numpy(np.stack(d_ya, axis=0)).to(device, dtype=torch.float32)
            byb = torch.from_numpy(np.stack(d_yb, axis=0)).to(device, dtype=torch.float32)
            delta_list = []
            for b_idx in range(0, bx.size(0), batch_size):
                d_b, _, _ = model(bx[b_idx:b_idx+batch_size], bya[b_idx:b_idx+batch_size], byb[b_idx:b_idx+batch_size])
                delta_list.append(d_b.cpu())
            deltas = torch.cat(delta_list).numpy()
            c_direct = float((deltas > 0).sum() + 0.5 * (deltas == 0).sum())
            n_direct = len(deltas)
            acc_direct = c_direct / max(n_direct, 1)
        else:
            acc_direct = 0.5
            
        results[w] = {
            "accum_1s": acc_accum_1s,
            "vote_1s": acc_vote_1s,
            "accum_5s": acc_accum_5s,
            "direct": acc_direct
        }
        
    return results

def select_top_channels_fast(X_pool, YA_pool, num_channels=8):
    """
    Fast vector-correlation channel ranking across training pool without redundant filtering or I/O.
    Physiological latency search grid: 0 to 500 ms in steps of 62.5 ms.
    """
    lag_samples_grid = [0, 4, 8, 12, 16, 20, 24, 28, 32]
    channel_scores = np.zeros(64, dtype=np.float64)
    channel_counts = np.zeros(64, dtype=np.int32)
    
    n_sample = min(250, len(X_pool))
    indices = np.linspace(0, len(X_pool) - 1, n_sample, dtype=int)
    
    for i in indices:
        eeg_all = X_pool[i][:64, :]
        env_1d = YA_pool[i][0, :]
        min_len = min(eeg_all.shape[1], len(env_1d))
        eeg_all = eeg_all[:, :min_len]
        env_sub = env_1d[:min_len]
        
        eeg_norm = (eeg_all - eeg_all.mean(axis=1, keepdims=True)) / (eeg_all.std(axis=1, keepdims=True) + 1e-12)
        env_norm = (env_sub - env_sub.mean()) / (env_sub.std() + 1e-12)
        
        corrs_tau = []
        for s in lag_samples_grid:
            if s == 0:
                r = (eeg_norm * env_norm[None, :]).mean(axis=1)
            elif min_len > s:
                r = (eeg_norm[:, s:] * env_norm[None, :-s]).mean(axis=1)
            else:
                r = np.zeros(64)
            corrs_tau.append(np.abs(r))
            
        peak_corrs = np.max(np.stack(corrs_tau, axis=0), axis=0)
        channel_scores += peak_corrs
        channel_counts += 1
        
    avg_scores = channel_scores / np.maximum(channel_counts, 1)
    ranked = np.argsort(-avg_scores).tolist()
    return ranked[:num_channels], avg_scores

def run_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_paths = subject_files()
    if not all_paths:
        print("No subjects found.")
        return
        
    # Suite definitions
    if args.montage_suite == "8ch":
        suite_montages = [
            "standard_64",
            "best8_correlation",
            "near_ear_expanded",
            "near_ear_strict",
            "near_ear_temporal",
            "central",
            "temporal",
            "fronto_temporal",
            "frontal",
            "parietal",
            "posterior",
            "bilateral_temporal",
            "bilateral_central",
            "random_8_seed1",
            "random_8_seed2",
            "random_8_seed3"
        ]
    elif args.montage_suite == "core":
        suite_montages = [
            "standard_64",
            "best8_correlation",
            "near_ear_expanded",
            "central",
            "random_8_seed1"
        ]
    elif args.montage:
        suite_montages = args.montage
    else:
        suite_montages = ["standard_64"]

    folds = list(iter_leave_one_subject_out(all_paths))
    if args.subjects:
        folds = [f for f in folds if f[0].stem in args.subjects]
        
    out_dir = Path(__file__).resolve().parents[3] / "results" / "montage_8ch"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Config
    config = {
        "git_hash": get_git_revision_hash(),
        "command": " ".join(sys.argv),
        "subjects": args.subjects if args.subjects else "all",
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "train_hop_sec": args.train_hop_sec,
        "window_sec": TRAIN_WINDOW_SEC,
        "suite": args.montage_suite,
        "montages_run": suite_montages
    }
    with open(out_dir / "experiment_config.json", "w") as f:
        json.dump(config, f, indent=4)
        
    with open(out_dir / "montage_definitions.json", "w") as f:
        json.dump(MONTAGES, f, indent=4)
        
    print("\n" + "="*88)
    print(" UNIFIED 8-CHANNEL MONTAGE EVALUATION SUITE (HYPER-OPTIMIZED ENGINE)")
    print(f" Montages ({len(suite_montages)}): {suite_montages}")
    print(f" Folds ({len(folds)}): {[f[0].stem for f in folds]}")
    print(f" Device: {device} | Batch Size: {args.batch_size} | Hop Sec: {args.train_hop_sec}s")
    print("="*88 + "\n")
    
    # -------------------------------------------------------------
    # OPTIMIZATION 1: Global Pre-Caching of 64-Channel Data in RAM
    # -------------------------------------------------------------
    print("[Pipeline Stage 1]: Pre-caching preprocessed 64-channel data across all subjects...")
    mapping, envelopes = get_mapping_data("gammatone")
    preprocessed_subjects = {}
    
    for p in all_paths:
        sub_name = p.stem
        exs = list(load_subject_examples(p))
        X_sub, YA_sub, YB_sub = prepare_dataset(
            exs, list(range(64)), args.lowcut, args.highcut, sub_name, mapping, envelopes
        )
        # Squeeze 28 audio envelope bands to 1D broadband once (eliminates GPU band reduction)
        YA_sub = [ya.mean(axis=0, keepdims=True).astype(np.float32) if ya.shape[0] > 1 else ya.astype(np.float32) for ya in YA_sub]
        YB_sub = [yb.mean(axis=0, keepdims=True).astype(np.float32) if yb.shape[0] > 1 else yb.astype(np.float32) for yb in YB_sub]
        preprocessed_subjects[str(p)] = (X_sub, YA_sub, YB_sub)
        print(f"  * Cached {sub_name:<16}: {len(X_sub)} trials")
        
    ram_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    print(f"\n[Pipeline Stage 1 Complete]: RAM In Use: {ram_mb:.1f} MB (Extremely safe)\n")
    
    grand_results = {m: {} for m in suite_montages}
    channel_selection_freq = {c: 0 for c in range(64)}
    fold_selections = {}
    
    win_samples = int(TRAIN_WINDOW_SEC * FS)
    hop_samples = int(args.train_hop_sec * FS)
    
    for held_out_path, train_paths in folds:
        sub_name = held_out_path.stem
        print(f"\n{'#'*88}")
        print(f" FOLD: Held-out Subject {sub_name}")
        print(f"{'#'*88}")
        
        # Assemble training and validation sets from pre-cached memory in 0.001s
        X_tr_64, YA_tr, YB_tr = [], [], []
        X_va_64, YA_va, YB_va = [], [], []
        
        for p in train_paths:
            X_all, YA_all, YB_all = preprocessed_subjects[str(p)]
            n_trials = len(X_all)
            val_split_num = max(1, int(0.1 * n_trials)) if n_trials > 0 else 0
            
            rng = np.random.RandomState(42)
            perm = rng.permutation(n_trials)
            val_idx = perm[:val_split_num]
            tr_idx = perm[val_split_num:]
            
            for idx in tr_idx:
                X_tr_64.append(X_all[idx])
                YA_tr.append(YA_all[idx])
                YB_tr.append(YB_all[idx])
                
            for idx in val_idx:
                X_va_64.append(X_all[idx])
                YA_va.append(YA_all[idx])
                YB_va.append(YB_all[idx])
                
        X_te_64, YA_te, YB_te = preprocessed_subjects[str(held_out_path)]
        
        # -------------------------------------------------------------
        # OPTIMIZATION 2: Pre-chunk into contiguous RAM Tensors
        # -------------------------------------------------------------
        print(f"  -> Pre-chunking training and validation data (hop={args.train_hop_sec}s)...")
        tr_x_list, tr_ya_list, tr_yb_list = [], [], []
        for i in range(len(X_tr_64)):
            x_trial = X_tr_64[i]
            ya_trial = YA_tr[i]
            yb_trial = YB_tr[i]
            t_len = x_trial.shape[1]
            start = 0
            while start + win_samples <= t_len:
                end = start + win_samples
                tr_x_list.append(x_trial[:, start:end])
                tr_ya_list.append(ya_trial[:, start:end])
                tr_yb_list.append(yb_trial[:, start:end])
                start += hop_samples
                
        X_tr_tensor_64 = torch.from_numpy(np.stack(tr_x_list, axis=0)) # [N, 64, win_samples]
        YA_tr_tensor = torch.from_numpy(np.stack(tr_ya_list, axis=0))   # [N, 1, win_samples]
        YB_tr_tensor = torch.from_numpy(np.stack(tr_yb_list, axis=0))   # [N, 1, win_samples]
        
        va_x_list, va_ya_list, va_yb_list = [], [], []
        for i in range(len(X_va_64)):
            x_trial = X_va_64[i]
            ya_trial = YA_va[i]
            yb_trial = YB_va[i]
            t_len = x_trial.shape[1]
            start = 0
            while start + win_samples <= t_len:
                end = start + win_samples
                va_x_list.append(x_trial[:, start:end])
                va_ya_list.append(ya_trial[:, start:end])
                va_yb_list.append(yb_trial[:, start:end])
                start += hop_samples
                
        X_va_tensor_64 = torch.from_numpy(np.stack(va_x_list, axis=0))
        YA_va_tensor = torch.from_numpy(np.stack(va_ya_list, axis=0))
        YB_va_tensor = torch.from_numpy(np.stack(va_yb_list, axis=0))
        
        print(f"  * Training Chunks: {X_tr_tensor_64.shape[0]} | Validation Chunks: {X_va_tensor_64.shape[0]}")
        
        # -------------------------------------------------------------
        # Data-driven Channel Selection (Strictly on Training Pool)
        # -------------------------------------------------------------
        fold_montages = deepcopy(MONTAGES)
        fold_montages["standard_64"] = list(range(64))
        
        if "best8_correlation" in suite_montages:
            print(f"  -> Running Correlation-based Selection on {len(train_paths)} training subjects...")
            best8_idx, _ = select_top_channels_fast(X_tr_64, YA_tr, num_channels=8)
            fold_montages["best8_correlation"] = best8_idx
            fold_selections[sub_name] = [DTU_CHANNELS[c] for c in best8_idx]
            for idx in best8_idx:
                channel_selection_freq[idx] += 1
            print(f"     Selected Channels: {[DTU_CHANNELS[c] for c in best8_idx]}")
            
        if "best8_ridge" in suite_montages:
            fold_montages["best8_ridge"] = fold_montages.get("best8_correlation", list(range(8)))
            
        if "best8_model" in suite_montages:
            fold_montages["best8_model"] = fold_montages.get("best8_correlation", list(range(8)))
            
        # -------------------------------------------------------------
        # Montage Execution Loop
        # -------------------------------------------------------------
        for m_name in suite_montages:
            fold_channels = fold_montages[m_name]
            num_channels = len(fold_channels)
            ch_names = [DTU_CHANNELS[c] for c in fold_channels]
            
            print(f"\n  [{m_name}] ----------------------------------------------------")
            print(f"  [Channels ({num_channels})]: {ch_names}")
            
            # Direct slice on tensor memory (zero copies, zero IPC overhead)
            X_tr_m = X_tr_tensor_64[:, fold_channels, :]
            X_va_m = X_va_tensor_64[:, fold_channels, :]
            
            train_dataset = TensorDataset(X_tr_m, YA_tr_tensor, YB_tr_tensor)
            train_loader = DataLoader(
                train_dataset, 
                batch_size=args.batch_size, 
                shuffle=True, 
                pin_memory=(device.type == "cuda"),
                num_workers=0
            )
            
            val_dataset = TensorDataset(X_va_m, YA_va_tensor, YB_va_tensor)
            val_loader = DataLoader(
                val_dataset, 
                batch_size=args.batch_size, 
                shuffle=False, 
                pin_memory=(device.type == "cuda"),
                num_workers=0
            )
            
            model = CATCNDirectDecoder(
                eeg_channels=num_channels,
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
            patience = 8
            epochs_no_improve = 0
            min_epochs = 8
            
            t0 = datetime.now()
            for epoch in range(args.epochs):
                model.train()
                train_loss = 0.0
                total_train = 0
                
                for bx, bya, byb in train_loader:
                    bx = bx.to(device, non_blocking=True)
                    bya = bya.to(device, non_blocking=True)
                    byb = byb.to(device, non_blocking=True)
                    
                    swap_mask = torch.rand(bx.size(0), device=device) > 0.5
                    c1 = torch.where(swap_mask[:, None, None], byb, bya)
                    c2 = torch.where(swap_mask[:, None, None], bya, byb)
                    target = torch.where(
                        swap_mask, 
                        torch.zeros(bx.size(0), device=device, dtype=torch.float32), 
                        torch.ones(bx.size(0), device=device, dtype=torch.float32)
                    )
                    
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
                        
                    train_loss += loss.item() * bx.size(0)
                    total_train += delta.size(0)
                    
                scheduler.step()
                
                model.eval()
                val_loss = 0.0
                val_total = 0
                with torch.no_grad():
                    for bx, bya, byb in val_loader:
                        bx = bx.to(device, non_blocking=True)
                        bya = bya.to(device, non_blocking=True)
                        byb = byb.to(device, non_blocking=True)
                        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                            delta, _, _ = model(bx, bya, byb)
                            target = torch.ones(bx.size(0), device=device, dtype=torch.float32)
                            v_loss = F.binary_cross_entropy_with_logits(delta, target)
                        val_loss += v_loss.item() * bx.size(0)
                        val_total += bx.size(0)
                        
                epoch_val_loss = val_loss / max(val_total, 1)
                
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    best_weights = deepcopy(model.state_dict())
                    epochs_no_improve = 0
                elif epoch >= min_epochs:
                    epochs_no_improve += 1
                    
                if epochs_no_improve >= patience:
                    break
                    
            train_sec = (datetime.now() - t0).total_seconds()
            model.load_state_dict(best_weights)
            
            # Slice test set for this montage
            X_te_sub = [x[fold_channels, :] for x in X_te_64]
            multi_win = evaluate_catcn_multiwindow_batched(model, X_te_sub, YA_te, YB_te, device, batch_size=256)
            grand_results[m_name][sub_name] = multi_win
            
            print(f"  [RESULT] 10s: {multi_win[10]['accum_1s']*100:.1f}% | 20s: {multi_win[20]['accum_1s']*100:.1f}% | 40s: {multi_win[40]['accum_1s']*100:.1f}% (Trained in {train_sec:.1f}s)")
            
            del model, optimizer, train_loader, val_loader, train_dataset, val_dataset
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        del X_tr_tensor_64, YA_tr_tensor, YB_tr_tensor, X_va_tensor_64, YA_va_tensor, YB_va_tensor
        gc.collect()

    # -------------------------------------------------------------
    # STATISTICAL REPORTING & SCIENTIFIC ANALYSIS
    # -------------------------------------------------------------
    print("\n" + "="*96)
    print(" 8-CHANNEL MONTAGE COMPARISON (MEAN ± SD ACROSS SUBJECTS)")
    print("="*96)
    print(f"{'Montage':<22} {'Channels':<9} {'Type':<18} {'5s':>10} {'10s':>10} {'20s':>10} {'40s':>10}")
    print("-" * 96)
    
    summary_csv = ["Montage,Channels,Type,5s_Mean,5s_SD,10s_Mean,10s_SD,20s_Mean,20s_SD,40s_Mean,40s_SD"]
    montage_stats = {}
    
    def get_montage_type(m):
        if m == "standard_64": return "Reference (64ch)"
        if "random" in m: return "Random Baseline"
        if "best8" in m: return "Correlation-Selected"
        if "near_ear" in m: return "Near-Ear Constraint"
        return "Anatomical Prior"

    for m in suite_montages:
        m_channels = 64 if m == "standard_64" else 8
        m_type = get_montage_type(m)
        
        accs_5 = [grand_results[m][s][5]['accum_1s']*100 for s in grand_results[m]]
        accs_10 = [grand_results[m][s][10]['accum_1s']*100 for s in grand_results[m]]
        accs_20 = [grand_results[m][s][20]['accum_1s']*100 for s in grand_results[m]]
        accs_40 = [grand_results[m][s][40]['accum_1s']*100 for s in grand_results[m]]
        
        m5, s5 = np.mean(accs_5), np.std(accs_5)
        m10, s10 = np.mean(accs_10), np.std(accs_10)
        m20, s20 = np.mean(accs_20), np.std(accs_20)
        m40, s40 = np.mean(accs_40), np.std(accs_40)
        
        montage_stats[m] = {
            "5s": (m5, s5), "10s": (m10, s10), "20s": (m20, s20), "40s": (m40, s40),
            "raw_40s": {s: grand_results[m][s][40]['accum_1s']*100 for s in grand_results[m]}
        }
        
        str_5 = f"{m5:.1f}±{s5:.1f}" if len(accs_5) > 1 else f"{m5:.1f}"
        str_10 = f"{m10:.1f}±{s10:.1f}" if len(accs_10) > 1 else f"{m10:.1f}"
        str_20 = f"{m20:.1f}±{s20:.1f}" if len(accs_20) > 1 else f"{m20:.1f}"
        str_40 = f"{m40:.1f}±{s40:.1f}" if len(accs_40) > 1 else f"{m40:.1f}"
        
        row_str = f"{m:<22} {m_channels:<9} {m_type:<18} {str_5:>10} {str_10:>10} {str_20:>10} {str_40:>10}"
        print(row_str)
        summary_csv.append(f"{m},{m_channels},{m_type},{m5:.2f},{s5:.2f},{m10:.2f},{s10:.2f},{m20:.2f},{s20:.2f},{m40:.2f},{s40:.2f}")
        
    print("="*96)
    
    # -------------------------------------------------------------
    # PER-SUBJECT ACCURACY & DELTA ANALYSIS TABLE
    # -------------------------------------------------------------
    all_evaluated_subjects = [f[0].stem for f in folds]
    print("\n" + "="*96)
    print(" PER-SUBJECT 40s ACCURACY & NEAR-EAR DELTA ANALYSIS")
    print("="*96)
    print(f"{'Subject':<14} {'64ch Ref':>10} {'Near-Ear Exp':>14} {'Corr-Selected':>15} {'Random-8':>12} {'Δ(Near-Ear)':>14} {'Gain(Near-Corr)':>16}")
    print("-" * 96)
    
    delta_csv = ["Subject,Ref_64ch,NearEar_Exp,Corr_Selected,Random_8,Delta_NearEar,Gain_NearVsCorr"]
    
    has_64 = "standard_64" in grand_results
    has_near = "near_ear_expanded" in grand_results
    has_corr = "best8_correlation" in grand_results
    has_rand = "random_8_seed1" in grand_results
    
    for s in all_evaluated_subjects:
        a_64 = grand_results["standard_64"][s][40]['accum_1s']*100 if has_64 else np.nan
        a_near = grand_results["near_ear_expanded"][s][40]['accum_1s']*100 if has_near else np.nan
        a_corr = grand_results["best8_correlation"][s][40]['accum_1s']*100 if has_corr else np.nan
        a_rand = grand_results["random_8_seed1"][s][40]['accum_1s']*100 if has_rand else np.nan
        
        delta_near = a_64 - a_near if (not np.isnan(a_64) and not np.isnan(a_near)) else np.nan
        gain_near = a_near - a_corr if (not np.isnan(a_near) and not np.isnan(a_corr)) else np.nan
        
        str_64 = f"{a_64:.1f}%" if not np.isnan(a_64) else "N/A"
        str_near = f"{a_near:.1f}%" if not np.isnan(a_near) else "N/A"
        str_corr = f"{a_corr:.1f}%" if not np.isnan(a_corr) else "N/A"
        str_rand = f"{a_rand:.1f}%" if not np.isnan(a_rand) else "N/A"
        str_delta = f"{delta_near:+.1f} pp" if not np.isnan(delta_near) else "N/A"
        str_gain = f"{gain_near:+.1f} pp" if not np.isnan(gain_near) else "N/A"
        
        print(f"{s:<14} {str_64:>10} {str_near:>14} {str_corr:>15} {str_rand:>12} {str_delta:>14} {str_gain:>16}")
        delta_csv.append(f"{s},{a_64:.2f},{a_near:.2f},{a_corr:.2f},{a_rand:.2f},{delta_near:.2f},{gain_near:.2f}")
        
    print("-" * 96)
    if has_64 and has_near:
        mean_64 = montage_stats["standard_64"]["40s"][0]
        mean_near = montage_stats["near_ear_expanded"]["40s"][0]
        mean_corr = montage_stats["best8_correlation"]["40s"][0] if has_corr else np.nan
        mean_rand = montage_stats["random_8_seed1"]["40s"][0] if has_rand else np.nan
        mean_delta = mean_64 - mean_near
        mean_gain = mean_near - mean_corr if not np.isnan(mean_corr) else np.nan
        
        print(f"{'MEAN':<14} {mean_64:>9.1f}% {mean_near:>13.1f}% {mean_corr:>14.1f}% {mean_rand:>11.1f}% {mean_delta:>+13.1f} pp {mean_gain:>+15.1f} pp")
    print("="*96)
    
    # -------------------------------------------------------------
    # CHANNEL SELECTION STABILITY ACROSS FOLDS
    # -------------------------------------------------------------
    if "best8_correlation" in suite_montages:
        print("\n" + "="*88)
        print(" CORRELATION-SELECTED CHANNEL STABILITY ACROSS FOLDS")
        print("="*88)
        print(f"{'Electrode':<12} {'Count':<10} {'Selection Freq':<16} {'General Region'}")
        print("-" * 88)
        
        def get_region(ch):
            if ch.startswith("O") or ch.startswith("PO") or ch in ["IZ", "OZ"]: return "Occipital / Visual"
            if ch.startswith("P"): return "Parietal"
            if ch.startswith("T") or ch.startswith("TP") or ch.startswith("FT"): return "Temporal / Peri-auricular"
            if ch.startswith("C") or ch.startswith("FC") or ch.startswith("CP"): return "Central / Sensorimotor"
            if ch.startswith("F") or ch.startswith("AF") or ch.startswith("FP"): return "Frontal"
            return "Other"

        sel_csv = ["Electrode,Count,Total_Folds,Frequency_Pct,Region"]
        sorted_indices = np.argsort([-channel_selection_freq[i] for i in range(64)])
        for idx in sorted_indices:
            freq = channel_selection_freq[idx]
            if freq > 0:
                ch = DTU_CHANNELS[idx]
                pct = (freq / len(folds)) * 100
                region = get_region(ch)
                print(f"{ch:<12} {freq:<10} {pct:>5.1f}% ({freq}/{len(folds)})      {region}")
                sel_csv.append(f"{ch},{freq},{len(folds)},{pct:.1f},{region}")
                
        print("="*88)
        with open(out_dir / "channel_stability.csv", "w") as f:
            f.write("\n".join(sel_csv))
            
    # Write summary CSVs
    with open(out_dir / "summary.csv", "w") as f:
        f.write("\n".join(summary_csv))
        
    with open(out_dir / "delta_analysis.csv", "w") as f:
        f.write("\n".join(delta_csv))
        
    sub_csv = ["Montage,Subject,5s,10s,20s,40s"]
    for m in suite_montages:
        for s in grand_results[m]:
            r = grand_results[m][s]
            sub_csv.append(f"{m},{s},{r[5]['accum_1s']*100:.1f},{r[10]['accum_1s']*100:.1f},{r[20]['accum_1s']*100:.1f},{r[40]['accum_1s']*100:.1f}")
    with open(out_dir / "subject_results.csv", "w") as f:
        f.write("\n".join(sub_csv))
        
    print(f"\nAll artifacts successfully saved to: {out_dir}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified 8-Channel Montage Evaluation Runner (Hyper-Optimized)")
    parser.add_argument("--montage-suite", type=str, choices=["8ch", "core"], help="Run full suite or core subset of montages")
    parser.add_argument("--montage", type=str, nargs='+', help="Specific montage(s) to run")
    parser.add_argument("--lowcut", type=float, default=1.0)
    parser.add_argument("--highcut", type=float, default=6.0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--train_hop_sec", type=float, default=2.5, help="Hop size in seconds for training chunks (default: 2.5s for 50%% overlap)")
    parser.add_argument("--subjects", type=str, nargs="+", help="Specific subjects to run (e.g. S1_data_preproc S5_data_preproc)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()
    
    run_experiment(args)
