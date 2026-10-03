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
def evaluate_catcn_trial_multiwindow(model, x_trial, ya_trial, yb_trial, device, batch_size=256):
    """
    Evaluates CA-TCN on a single held-out trial across 5s, 10s, 20s, and 40s decision windows.
    Returns dictionary of decision scores and binary correctness.
    """
    model.eval()
    samples_1s = int(1 * FS)
    trial_len = x_trial.shape[1]
    
    # 1. Extract 1-second atomic sub-windows
    c_1s_x, c_1s_ya, c_1s_yb = [], [], []
    start = 0
    while start + samples_1s <= trial_len:
        end = start + samples_1s
        c_1s_x.append(x_trial[:, start:end])
        c_1s_ya.append(ya_trial[:, start:end])
        c_1s_yb.append(yb_trial[:, start:end])
        start += samples_1s
        
    if not c_1s_x:
        return {w: 0.5 for w in [5, 10, 20, 40]}
        
    bx = torch.from_numpy(np.stack(c_1s_x, axis=0)).to(device, dtype=torch.float32)
    bya = torch.from_numpy(np.stack(c_1s_ya, axis=0)).to(device, dtype=torch.float32)
    byb = torch.from_numpy(np.stack(c_1s_yb, axis=0)).to(device, dtype=torch.float32)
    
    delta_list = []
    for b_idx in range(0, bx.size(0), batch_size):
        d_b, _, _ = model(bx[b_idx:b_idx+batch_size], bya[b_idx:b_idx+batch_size], byb[b_idx:b_idx+batch_size])
        delta_list.append(d_b.detach().cpu())
    deltas = torch.cat(delta_list).numpy()
    d_1s = deltas.tolist()
    
    results = {}
    for w in [5, 10, 20, 40]:
        c_accum, n_accum = 0.0, 0
        b_start = 0
        while b_start + w <= len(d_1s):
            block_d = d_1s[b_start : b_start + w]
            if sum(block_d) > 0:
                c_accum += 1.0
            elif sum(block_d) == 0:
                c_accum += 0.5
            n_accum += 1
            b_start += w
        results[w] = (c_accum / max(n_accum, 1)) if n_accum > 0 else 0.5
        
    return results

def select_top_channels_fast(X_pool, YA_pool, num_channels=8):
    """
    Fast vector-correlation channel ranking across training pool without test leakage.
    Physiological latency search grid: 0 to 500 ms in steps of 62.5 ms.
    """
    lag_samples_grid = [0, 4, 8, 12, 16, 20, 24, 28, 32]
    channel_scores = np.zeros(64, dtype=np.float64)
    channel_counts = np.zeros(64, dtype=np.int32)
    
    n_sample = min(200, len(X_pool))
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

def run_loto_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_paths = subject_files()
    if not all_paths:
        print("No subjects found.")
        return
        
    if args.subjects:
        subjects_to_run = [p for p in all_paths if p.stem in args.subjects]
    else:
        subjects_to_run = all_paths
        
    out_dir = Path(__file__).resolve().parents[3] / "results" / "loto"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    suite_montages = args.montage if args.montage else ["standard_64", "near_ear_expanded"]
    
    print("\n" + "="*88)
    print(" CA-TCN LEAVE-ONE-TRIAL-OUT (LOTO) WITHIN-SUBJECT EXPERIMENT RUNNER")
    print(f" Protocol: {args.protocol.upper()}")
    print(f" Subjects to Evaluate ({len(subjects_to_run)}): {[p.stem for p in subjects_to_run]}")
    print(f" Montages ({len(suite_montages)}): {suite_montages}")
    print(f" Device: {device} | Epochs per Fold: {args.epochs} | Batch Size: {args.batch_size}")
    print("="*88 + "\n")
    
    mapping, envelopes = get_mapping_data("gammatone")
    
    win_samples = int(TRAIN_WINDOW_SEC * FS)
    hop_samples = int(args.train_hop_sec * FS)
    
    grand_subject_results = {m: {} for m in suite_montages}
    
    # Check if prior LOSO results exist for direct delta comparison
    loso_summary = {}
    loso_file = Path(__file__).resolve().parents[3] / "results" / "montage_8ch" / "subject_results.csv"
    if loso_file.exists():
        try:
            with open(loso_file, "r") as f:
                lines = f.readlines()[1:]
                for l in lines:
                    parts = l.strip().split(",")
                    if len(parts) >= 6:
                        m_name, sub, a5, a10, a20, a40 = parts[0], parts[1], float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                        if m_name not in loso_summary: loso_summary[m_name] = {}
                        loso_summary[m_name][sub] = {"5s": a5, "10s": a10, "20s": a20, "40s": a40}
            print(f"[LOSO Linkage]: Successfully loaded prior LOSO results from {loso_file}\n")
        except Exception as e:
            print(f"[LOSO Linkage Warning]: Could not parse LOSO file: {e}")
            
    # Iterate across subjects
    for sub_idx, sub_path in enumerate(subjects_to_run):
        sub_name = sub_path.stem
        sub_dir = out_dir / sub_name
        sub_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'#'*88}")
        print(f" SUBJECT ({sub_idx+1}/{len(subjects_to_run)}): {sub_name}")
        print(f"{'#'*88}")
        
        # Load & Preprocess subject data once (64 channels)
        exs = list(load_subject_examples(sub_path))
        X_sub, YA_sub, YB_sub = prepare_dataset(
            exs, list(range(64)), args.lowcut, args.highcut, sub_name, mapping, envelopes
        )
        YA_sub = [ya.mean(axis=0, keepdims=True).astype(np.float32) if ya.shape[0] > 1 else ya.astype(np.float32) for ya in YA_sub]
        YB_sub = [yb.mean(axis=0, keepdims=True).astype(np.float32) if yb.shape[0] > 1 else yb.astype(np.float32) for yb in YB_sub]
        
        n_trials = len(X_sub)
        print(f"  * Total Available Trials: {n_trials}")
        
        # Determine trial grouping based on protocol
        if args.protocol == "leave_story_out":
            # Group trials by audio story name
            story_groups = {}
            for t_i in range(n_trials):
                tr_key = f"trial_{t_i}"
                sub_key = sub_name.replace("_data_preproc", "")
                if sub_key in mapping and tr_key in mapping[sub_key]:
                    fname = mapping[sub_key][tr_key]["wavA"]["filename"]
                    story_id = "_".join(fname.split("_")[:2]) # e.g. marianne_story3
                else:
                    story_id = f"trial_{t_i}"
                if story_id not in story_groups: story_groups[story_id] = []
                story_groups[story_id].append(t_i)
            fold_partitions = list(story_groups.values())
            print(f"  * Leave-Story-Out Mode: {len(fold_partitions)} distinct story partitions")
        else:
            # Standard LOTO: 1 trial per fold
            fold_partitions = [[i] for i in range(n_trials)]
            print(f"  * Standard LOTO Mode: {len(fold_partitions)} folds (59 train -> 1 test)")
            
        # Pre-chunk all trials into contiguous memory
        trial_chunk_slices = []
        all_chunks_x, all_chunks_ya, all_chunks_yb = [], [], []
        chunk_counter = 0
        
        for t_i in range(n_trials):
            x_t = X_sub[t_i]
            ya_t = YA_sub[t_i]
            yb_t = YB_sub[t_i]
            t_len = x_t.shape[1]
            c_indices = []
            start = 0
            while start + win_samples <= t_len:
                end = start + win_samples
                all_chunks_x.append(x_t[:, start:end])
                all_chunks_ya.append(ya_t[:, start:end])
                all_chunks_yb.append(yb_t[:, start:end])
                c_indices.append(chunk_counter)
                chunk_counter += 1
                start += hop_samples
            trial_chunk_slices.append(c_indices)
            
        X_all_tensor_64 = torch.from_numpy(np.stack(all_chunks_x, axis=0)) # [N, 64, win_samples]
        YA_all_tensor = torch.from_numpy(np.stack(all_chunks_ya, axis=0))   # [N, 1, win_samples]
        YB_all_tensor = torch.from_numpy(np.stack(all_chunks_yb, axis=0))   # [N, 1, win_samples]
        
        # Iterate through requested montages
        for m_name in suite_montages:
            print(f"\n  [{m_name}] Running {len(fold_partitions)} LOTO folds...")
            
            fold_results = {w: [] for w in [5, 10, 20, 40]}
            fold_records = []
            
            t0 = datetime.now()
            
            for fold_idx, held_out_trials in enumerate(fold_partitions):
                # 1. Identify training trials
                train_trials = [t for t in range(n_trials) if t not in held_out_trials]
                
                # 2. Gather training chunks
                train_chunk_idx = []
                for t in train_trials:
                    train_chunk_idx.extend(trial_chunk_slices[t])
                    
                # 3. Channel Selection (Strictly on training trials if data-driven)
                if m_name == "standard_64":
                    fold_channels = list(range(64))
                elif m_name == "best8_correlation":
                    X_tr_pool = [X_sub[t] for t in train_trials]
                    YA_tr_pool = [YA_sub[t] for t in train_trials]
                    fold_channels, _ = select_top_channels_fast(X_tr_pool, YA_tr_pool, num_channels=8)
                else:
                    fold_channels = MONTAGES[m_name]
                    
                # 4. Assemble fast TensorDataset in RAM
                X_tr_fold = X_all_tensor_64[train_chunk_idx][:, fold_channels, :]
                YA_tr_fold = YA_all_tensor[train_chunk_idx]
                YB_tr_fold = YB_all_tensor[train_chunk_idx]
                
                # 10% validation split from training chunks for early stopping
                n_tr_chunks = len(train_chunk_idx)
                n_val_split = max(1, int(0.1 * n_tr_chunks))
                
                perm = np.random.RandomState(42).permutation(n_tr_chunks)
                val_idx = perm[:n_val_split]
                actual_tr_idx = perm[n_val_split:]
                
                train_ds = TensorDataset(X_tr_fold[actual_tr_idx], YA_tr_fold[actual_tr_idx], YB_tr_fold[actual_tr_idx])
                val_ds = TensorDataset(X_tr_fold[val_idx], YA_tr_fold[val_idx], YB_tr_fold[val_idx])
                
                train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=0)
                val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=0)
                
                # 5. Initialize CA-TCN
                model = CATCNDirectDecoder(
                    eeg_channels=len(fold_channels),
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
                
                # 6. Train model on within-subject fold (blazing fast ~0.3s)
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
                
                # 7. Evaluate on held-out test trial(s)
                trial_res_list = []
                for test_t in held_out_trials:
                    x_te_m = X_sub[test_t][fold_channels, :]
                    ya_te = YA_sub[test_t]
                    yb_te = YB_sub[test_t]
                    t_win_res = evaluate_catcn_trial_multiwindow(model, x_te_m, ya_te, yb_te, device)
                    trial_res_list.append(t_win_res)
                    
                # Aggregate held-out performance for this fold
                avg_fold_res = {w: np.mean([r[w] for r in trial_res_list]) for w in [5, 10, 20, 40]}
                for w in [5, 10, 20, 40]:
                    fold_results[w].append(avg_fold_res[w])
                    
                # Save fold record
                rec = {
                    "subject": sub_name,
                    "fold": fold_idx,
                    "held_out_trials": held_out_trials,
                    "montage": m_name,
                    "channels": [DTU_CHANNELS[c] for c in fold_channels],
                    "acc_5s": avg_fold_res[5],
                    "acc_10s": avg_fold_res[10],
                    "acc_20s": avg_fold_res[20],
                    "acc_40s": avg_fold_res[40]
                }
                fold_records.append(rec)
                
                del model, optimizer, train_loader, val_loader, train_ds, val_ds
                if fold_idx % 20 == 0 or fold_idx == len(fold_partitions) - 1:
                    print(f"     Fold {fold_idx+1:>2}/{len(fold_partitions)} -> Held-out Trial(s): {held_out_trials} | 40s Acc: {avg_fold_res[40]*100:.1f}%")
                    
            elapsed_sec = (datetime.now() - t0).total_seconds()
            
            # Save fold-level JSON records
            with open(sub_dir / f"folds_{m_name}.json", "w") as f:
                json.dump(fold_records, f, indent=2)
                
            mean_5 = np.mean(fold_results[5]) * 100
            mean_10 = np.mean(fold_results[10]) * 100
            mean_20 = np.mean(fold_results[20]) * 100
            mean_40 = np.mean(fold_results[40]) * 100
            
            grand_subject_results[m_name][sub_name] = {
                "5s": mean_5, "10s": mean_10, "20s": mean_20, "40s": mean_40
            }
            
            print(f"  [SUMMARY {m_name}]: 10s: {mean_10:.1f}% | 20s: {mean_20:.1f}% | 40s: {mean_40:.1f}% (All {len(fold_partitions)} folds finished in {elapsed_sec:.1f}s)")
            
        del X_all_tensor_64, YA_all_tensor, YB_all_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # GRAND LOTO SUMMARY TABLE
    # -------------------------------------------------------------
    print("\n" + "="*96)
    print(f" LEAVE-ONE-TRIAL-OUT (LOTO) SUMMARY TABLE ({args.protocol.upper()})")
    print("="*96)
    print(f"{'Montage':<22} {'Channels':<9} {'Subjects':<10} {'5s':>10} {'10s':>10} {'20s':>10} {'40s':>10}")
    print("-" * 96)
    
    summary_csv = ["Montage,Channels,Protocol,5s_Mean,10s_Mean,20s_Mean,40s_Mean"]
    
    for m in suite_montages:
        m_channels = 64 if m == "standard_64" else 8
        m5_arr = [grand_subject_results[m][s]["5s"] for s in grand_subject_results[m]]
        m10_arr = [grand_subject_results[m][s]["10s"] for s in grand_subject_results[m]]
        m20_arr = [grand_subject_results[m][s]["20s"] for s in grand_subject_results[m]]
        m40_arr = [grand_subject_results[m][s]["40s"] for s in grand_subject_results[m]]
        
        m5, s5 = np.mean(m5_arr), np.std(m5_arr)
        m10, s10 = np.mean(m10_arr), np.std(m10_arr)
        m20, s20 = np.mean(m20_arr), np.std(m20_arr)
        m40, s40 = np.mean(m40_arr), np.std(m40_arr)
        
        str_5 = f"{m5:.1f}±{s5:.1f}" if len(m5_arr) > 1 else f"{m5:.1f}"
        str_10 = f"{m10:.1f}±{s10:.1f}" if len(m10_arr) > 1 else f"{m10:.1f}"
        str_20 = f"{m20:.1f}±{s20:.1f}" if len(m20_arr) > 1 else f"{m20:.1f}"
        str_40 = f"{m40:.1f}±{s40:.1f}" if len(m40_arr) > 1 else f"{m40:.1f}"
        
        print(f"{m:<22} {m_channels:<9} {len(subjects_to_run):<10} {str_5:>10} {str_10:>10} {str_20:>10} {str_40:>10}")
        summary_csv.append(f"{m},{m_channels},{args.protocol},{m5:.2f},{m10:.2f},{m20:.2f},{m40:.2f}")
    print("="*96)
    
    # -------------------------------------------------------------
    # DIRECT COMPARISON: LOSO (Cross-Subject) vs LOTO (Within-Subject)
    # -------------------------------------------------------------
    has_loso_64 = "standard_64" in loso_summary
    has_loso_near = "near_ear_expanded" in loso_summary
    has_loto_64 = "standard_64" in grand_subject_results
    has_loto_near = "near_ear_expanded" in grand_subject_results
    
    if (has_loso_64 or has_loso_near) and (has_loto_64 or has_loto_near):
        print("\n" + "="*104)
        print(" CROSS-SUBJECT (LOSO) vs WITHIN-SUBJECT (LOTO) 40s COMPARISON")
        print("="*104)
        print(f"{'Subject':<14} {'LOSO 64ch':>12} {'LOTO 64ch':>12} {'Δ(Within 64)':>15} {'LOSO Near-Ear':>15} {'LOTO Near-Ear':>15} {'Δ(Within Near)':>16}")
        print("-" * 104)
        
        comp_csv = ["Subject,LOSO_64ch,LOTO_64ch,Delta_Within_64,LOSO_NearEar,LOTO_NearEar,Delta_Within_NearEar"]
        
        delta_64_list = []
        delta_near_list = []
        
        for p in subjects_to_run:
            s = p.stem
            a_loso_64 = loso_summary.get("standard_64", {}).get(s, {}).get("40s", np.nan)
            a_loto_64 = grand_subject_results.get("standard_64", {}).get(s, {}).get("40s", np.nan)
            
            a_loso_near = loso_summary.get("near_ear_expanded", {}).get(s, {}).get("40s", np.nan)
            a_loto_near = grand_subject_results.get("near_ear_expanded", {}).get(s, {}).get("40s", np.nan)
            
            d_64 = a_loto_64 - a_loso_64 if (not np.isnan(a_loso_64) and not np.isnan(a_loto_64)) else np.nan
            d_near = a_loto_near - a_loso_near if (not np.isnan(a_loso_near) and not np.isnan(a_loto_near)) else np.nan
            
            if not np.isnan(d_64): delta_64_list.append(d_64)
            if not np.isnan(d_near): delta_near_list.append(d_near)
            
            str_lo64 = f"{a_loso_64:.1f}%" if not np.isnan(a_loso_64) else "N/A"
            str_lt64 = f"{a_loto_64:.1f}%" if not np.isnan(a_loto_64) else "N/A"
            str_d64 = f"{d_64:+.1f} pp" if not np.isnan(d_64) else "N/A"
            
            str_lonear = f"{a_loso_near:.1f}%" if not np.isnan(a_loso_near) else "N/A"
            str_ltnear = f"{a_loto_near:.1f}%" if not np.isnan(a_loto_near) else "N/A"
            str_dnear = f"{d_near:+.1f} pp" if not np.isnan(d_near) else "N/A"
            
            print(f"{s:<14} {str_lo64:>12} {str_lt64:>12} {str_d64:>15} {str_lonear:>15} {str_ltnear:>15} {str_dnear:>16}")
            comp_csv.append(f"{s},{a_loso_64:.2f},{a_loto_64:.2f},{d_64:.2f},{a_loso_near:.2f},{a_loto_near:.2f},{d_near:.2f}")
            
        print("-" * 104)
        mean_d64 = np.mean(delta_64_list) if delta_64_list else np.nan
        mean_dnear = np.mean(delta_near_list) if delta_near_list else np.nan
        print(f"{'MEAN':<14} {'':>12} {'':>12} {mean_d64:>+14.1f} pp {'':>15} {'':>15} {mean_dnear:>+15.1f} pp")
        print("="*104)
        
        with open(out_dir / "loso_vs_loto_comparison.csv", "w") as f:
            f.write("\n".join(comp_csv))
            
    with open(out_dir / "summary.csv", "w") as f:
        f.write("\n".join(summary_csv))
        
    print(f"\nAll LOTO artifacts saved to: {out_dir}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CA-TCN Leave-One-Trial-Out (LOTO) Within-Subject Runner")
    parser.add_argument("--subjects", type=str, nargs="+", help="Subjects to evaluate (e.g. S1_data_preproc)")
    parser.add_argument("--montage", type=str, nargs="+", help="Montages to evaluate (default: standard_64 near_ear_expanded)")
    parser.add_argument("--protocol", type=str, choices=["loto", "leave_story_out"], default="loto", help="Evaluation protocol")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per LOTO fold (default: 10)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for within-subject chunks")
    parser.add_argument("--train_hop_sec", type=float, default=2.5)
    parser.add_argument("--lowcut", type=float, default=1.0)
    parser.add_argument("--highcut", type=float, default=6.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()
    
    run_loto_experiment(args)
