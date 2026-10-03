import argparse
import sys
import os
import json
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
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
    FS, TRAIN_WINDOW_SEC, TRAIN_HOP_SEC,
    prepare_dataset, select_top_channels_from_train, get_mapping_data, ChunkDataset
)
from training.montages import MONTAGES, DTU_CHANNELS

def get_git_revision_hash() -> str:
    try:
        project_root = Path(__file__).resolve().parents[3]
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(project_root)).decode('ascii').strip()
    except Exception:
        return "unknown"

def evaluate_catcn_multiwindow(model, X, Y_A, Y_B, device):
    """
    Evaluates CA-TCN across multi-window decision lengths.
    """
    model.eval()
    
    trial_sub_1s = []
    trial_sub_5s = []
    
    samples_1s = int(1 * FS)
    samples_5s = int(5 * FS)
    
    with torch.no_grad():
        for i in range(len(X)):
            x_np, ya_np, yb_np = X[i], Y_A[i], Y_B[i]
            trial_len = x_np.shape[1]
            
            # --- 1-second atomic sub-windows ---
            d_1s, v_1s = [], []
            start = 0
            while start + samples_1s <= trial_len:
                end = start + samples_1s
                x_chunk = torch.from_numpy(x_np[:, start:end]).unsqueeze(0).to(device, dtype=torch.float32)
                ya_chunk = torch.from_numpy(ya_np[:, start:end]).unsqueeze(0).to(device, dtype=torch.float32)
                yb_chunk = torch.from_numpy(yb_np[:, start:end]).unsqueeze(0).to(device, dtype=torch.float32)
                
                delta, (la, lb), _ = model(x_chunk, ya_chunk, yb_chunk)
                diff = delta.item()
                d_1s.append(diff)
                v_1s.append(1.0 if diff > 0 else (0.5 if diff == 0 else 0.0))
                start += samples_1s
            trial_sub_1s.append((d_1s, v_1s))
            
            # --- 5-second atomic sub-windows ---
            d_5s, v_5s = [], []
            start = 0
            while start + samples_5s <= trial_len:
                end = start + samples_5s
                x_chunk = torch.from_numpy(x_np[:, start:end]).unsqueeze(0).to(device, dtype=torch.float32)
                ya_chunk = torch.from_numpy(ya_np[:, start:end]).unsqueeze(0).to(device, dtype=torch.float32)
                yb_chunk = torch.from_numpy(yb_np[:, start:end]).unsqueeze(0).to(device, dtype=torch.float32)
                
                delta, (la, lb), _ = model(x_chunk, ya_chunk, yb_chunk)
                diff = delta.item()
                d_5s.append(diff)
                v_5s.append(1.0 if diff > 0 else (0.5 if diff == 0 else 0.0))
                start += samples_5s
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
            
        c_direct, n_direct = 0.0, 0
        w_samples = int(w * FS)
        with torch.no_grad():
            for i in range(len(X)):
                x_np, ya_np, yb_np = X[i], Y_A[i], Y_B[i]
                start = 0
                while start + w_samples <= x_np.shape[1]:
                    end = start + w_samples
                    x_chunk = torch.from_numpy(x_np[:, start:end]).unsqueeze(0).to(device, dtype=torch.float32)
                    ya_chunk = torch.from_numpy(ya_np[:, start:end]).unsqueeze(0).to(device, dtype=torch.float32)
                    yb_chunk = torch.from_numpy(yb_np[:, start:end]).unsqueeze(0).to(device, dtype=torch.float32)
                    delta, _, _ = model(x_chunk, ya_chunk, yb_chunk)
                    if delta.item() > 0: c_direct += 1.0
                    elif delta.item() == 0: c_direct += 0.5
                    n_direct += 1
                    start += w_samples
        acc_direct = c_direct / max(n_direct, 1)
        
        results[w] = {
            "accum_1s": acc_accum_1s,
            "vote_1s": acc_vote_1s,
            "accum_5s": acc_accum_5s,
            "direct": acc_direct,
            "decisions": n_accum_1s
        }
    return results

def run_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_paths = subject_files()
    if not all_paths:
        print("No subjects found.")
        return
        
    # Suite definitions
    suite_montages = []
    if args.montage_suite == "8ch":
        suite_montages = [
            "standard_64",
            "best8_correlation",
            "frontal",
            "fronto_temporal",
            "temporal",
            "central",
            "parietal",
            "posterior",
            "bilateral_temporal",
            "bilateral_central",
            "near_ear_strict",
            "near_ear_expanded",
            "near_ear_temporal"
        ]
    elif args.montage:
        suite_montages = [args.montage]
    else:
        suite_montages = ["standard_64"]

    subject_examples = {str(p): load_subject_examples(p) for p in all_paths}
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
        "window_sec": TRAIN_WINDOW_SEC,
        "overlap_sec": TRAIN_WINDOW_SEC - TRAIN_HOP_SEC,
        "suite": args.montage_suite,
        "montages_run": suite_montages
    }
    with open(out_dir / "experiment_config.json", "w") as f:
        json.dump(config, f, indent=4)
        
    # Save Mapping
    with open(out_dir / "montage_definitions.json", "w") as f:
        json.dump(MONTAGES, f, indent=4)
        
    print("\n" + "="*88)
    print(f" UNIFIED 8-CHANNEL MONTAGE EVALUATION SUITE")
    print(f" Montages: {len(suite_montages)}")
    print(f" Folds: {len(folds)}")
    print(f" Device: {device}")
    print("="*88 + "\n")
    
    # Check 1: Authoritative Map Exists
    print("Check 1: Authoritative 64-Channel DTU map available.")
    
    # Check 2 & 3: Valid indices and names
    for m in suite_montages:
        if m in MONTAGES:
            indices = MONTAGES[m]
            assert len(indices) == 8, f"Montage {m} must have exactly 8 channels."
            for idx in indices:
                assert idx in DTU_CHANNELS, f"Invalid index {idx} in montage {m}"
                
    grand_results = {m: {} for m in suite_montages}
    channel_selection_freq = {c: 0 for c in range(64)}
    
    for held_out_path, train_paths in folds:
        sub_name = held_out_path.stem
        print(f"\n{'#'*88}")
        print(f" FOLD: Held-out Subject {sub_name}")
        print(f"{'#'*88}")
        
        mapping, envelopes = get_mapping_data("gammatone")
        
        # Cache 64-channel data to avoid redundant loading per montage
        print(f"  -> Pre-loading full 64-channel data for fold (Train: {len(train_paths)} subjects)...")
        
        X_tr_64, YA_tr, YB_tr = [], [], []
        X_va_64, YA_va, YB_va = [], [], []
        
        for p in train_paths:
            exs = list(subject_examples[str(p)])
            np.random.seed(42) # Consistent val split
            np.random.shuffle(exs)
            val_split_num = max(1, int(0.1 * len(exs))) if len(exs) > 0 else 0
            val_exs = exs[:val_split_num]
            train_exs = exs[val_split_num:]
            
            tX, tYA, tYB = prepare_dataset(train_exs, list(range(64)), args.lowcut, args.highcut, p.stem, mapping, envelopes)
            X_tr_64.extend(tX); YA_tr.extend(tYA); YB_tr.extend(tYB)
            
            vX, vYA, vYB = prepare_dataset(val_exs, list(range(64)), args.lowcut, args.highcut, p.stem, mapping, envelopes)
            X_va_64.extend(vX); YA_va.extend(vYA); YB_va.extend(vYB)
            
        test_exs = subject_examples[str(held_out_path)]
        X_te_64, YA_te, YB_te = prepare_dataset(test_exs, list(range(64)), args.lowcut, args.highcut, sub_name, mapping, envelopes)
        
        # Precompute data-driven montages for this fold using training subjects ONLY
        fold_montages = deepcopy(MONTAGES)
        fold_montages["standard_64"] = list(range(64))
        
        if "best8_correlation" in suite_montages:
            print(f"  -> Running Data-Driven Selection (Correlation) on {len(train_paths)} training subjects...")
            best8_idx, _ = select_top_channels_from_train(
                subject_examples, train_paths, mapping, envelopes, 8, args.lowcut, args.highcut
            )
            fold_montages["best8_correlation"] = best8_idx
            for idx in best8_idx:
                channel_selection_freq[idx] += 1
                
        if "best8_ridge" in suite_montages:
            # Fallback to correlation if true ridge selector is missing
            print("  -> best8_ridge requested, falling back to correlation for now.")
            fold_montages["best8_ridge"] = fold_montages.get("best8_correlation", list(range(8)))
            
        if "best8_model" in suite_montages:
            print("  -> best8_model requested, falling back to correlation for now.")
            fold_montages["best8_model"] = fold_montages.get("best8_correlation", list(range(8)))

        # Run each requested montage
        for m_name in suite_montages:
            print(f"\n  [{m_name}] ----------------------------------------------------")
            fold_channels = fold_montages[m_name]
            num_channels = len(fold_channels)
            ch_names = [DTU_CHANNELS[c] for c in fold_channels]
            
            print(f"  [Channel Setup]: {num_channels} ch -> {ch_names}")
            
            # Slice 64-channel data down to specific montage
            X_tr_sub = [x[fold_channels, :] for x in X_tr_64]
            X_va_sub = [x[fold_channels, :] for x in X_va_64]
            X_te_sub = [x[fold_channels, :] for x in X_te_64]
            
            # Convert to tensors
            X_tr_t = [torch.from_numpy(x) for x in X_tr_sub]
            YA_tr_t = [torch.from_numpy(x) for x in YA_tr]
            YB_tr_t = [torch.from_numpy(x) for x in YB_tr]
            
            win_samples = int(TRAIN_WINDOW_SEC * FS)
            hop_samples = int(1.25 * FS)
            
            chunk_indices = []
            for i in range(len(X_tr_t)):
                trial_len = X_tr_t[i].shape[1]
                start = 0
                while start + win_samples <= trial_len:
                    chunk_indices.append((i, start, start + win_samples))
                    start += hop_samples
                    
            train_dataset = ChunkDataset(X_tr_t, YA_tr_t, YB_tr_t, chunk_indices)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            
            val_chunk_indices = []
            for i in range(len(X_va_sub)):
                trial_len = X_va_sub[i].shape[1]
                start = 0
                while start + win_samples <= trial_len:
                    val_chunk_indices.append((i, start, start + win_samples))
                    start += hop_samples
                    
            X_va_t = [torch.from_numpy(x) for x in X_va_sub]
            YA_va_t = [torch.from_numpy(x) for x in YA_va]
            YB_va_t = [torch.from_numpy(x) for x in YB_va]
                    
            val_dataset = ChunkDataset(X_va_t, YA_va_t, YB_va_t, val_chunk_indices)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            
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
            patience = 10
            epochs_no_improve = 0
            min_epochs = 10
            
            for epoch in range(args.epochs):
                model.train()
                train_loss = 0.0
                correct_train = 0.0
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
                    
                    optimizer.zero_grad()
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
                    correct_train += ((delta > 0).float() == target).sum().item()
                    total_train += delta.size(0)
                    
                scheduler.step()
                train_epoch_loss = train_loss / total_train
                
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
                    
            model.load_state_dict(best_weights)
            multi_win = evaluate_catcn_multiwindow(model, X_te_sub, YA_te, YB_te, device)
            grand_results[m_name][sub_name] = multi_win
            
            # Print immediate evaluation
            print(f"  [RESULT] 10s: {multi_win[10]['accum_1s']*100:.1f}% | 20s: {multi_win[20]['accum_1s']*100:.1f}% | 40s: {multi_win[40]['accum_1s']*100:.1f}%")
            
            del model, optimizer, train_dataset, train_loader, val_dataset, val_loader
            del X_tr_t, X_va_t, YA_tr_t, YB_tr_t, YA_va_t, YB_va_t, X_tr_sub, X_va_sub, X_te_sub
            torch.cuda.empty_cache()
            gc.collect()

        # Clean fold 64-ch arrays
        del X_tr_64, X_va_64, X_te_64, YA_tr, YB_tr, YA_va, YB_va, YA_te, YB_te
        torch.cuda.empty_cache()
        gc.collect()

    # Generate Grand Summary
    print("\n" + "="*88)
    print("                 8-CHANNEL MONTAGE COMPARISON")
    print("="*88)
    print(f"{'Montage':<28} {'Channels':<14} {'5s':>8} {'10s':>8} {'20s':>8} {'40s':>8}")
    print("-" * 88)
    
    summary_csv = ["Montage,Channels,5s,10s,20s,40s"]
    montage_means = {}
    
    for m in suite_montages:
        if len(grand_results[m]) == 0:
            continue
            
        m_channels = 64 if m == "standard_64" else 8
        
        accs_5 = [grand_results[m][s][5]['accum_1s']*100 for s in grand_results[m]]
        accs_10 = [grand_results[m][s][10]['accum_1s']*100 for s in grand_results[m]]
        accs_20 = [grand_results[m][s][20]['accum_1s']*100 for s in grand_results[m]]
        accs_40 = [grand_results[m][s][40]['accum_1s']*100 for s in grand_results[m]]
        
        m5 = np.mean(accs_5)
        m10 = np.mean(accs_10)
        m20 = np.mean(accs_20)
        m40 = np.mean(accs_40)
        
        montage_means[m] = {"10s": m10, "20s": m20, "40s": m40}
        
        row_str = f"{m:<28} {m_channels:<14} {m5:>8.1f} {m10:>8.1f} {m20:>8.1f} {m40:>8.1f}"
        print(row_str)
        summary_csv.append(f"{m},{m_channels},{m5:.1f},{m10:.1f},{m20:.1f},{m40:.1f}")
        
    print("="*88)
    
    # Subject-level Print
    for m in suite_montages:
        if len(grand_results[m]) == 0:
            continue
        print("\n========================================================================================")
        print("SUBJECT-LEVEL RESULTS")
        print("========================================================================================")
        print(f"\nMontage: {m}\n")
        print(f"{'Subject':<14} {'5s':>8} {'10s':>8} {'20s':>8} {'40s':>8}")
        print("-" * 50)
        for s in grand_results[m]:
            r = grand_results[m][s]
            print(f"{s:<14} {r[5]['accum_1s']*100:>8.1f} {r[10]['accum_1s']*100:>8.1f} {r[20]['accum_1s']*100:>8.1f} {r[40]['accum_1s']*100:>8.1f}")
        print("-" * 50)
        accs_5_arr = [grand_results[m][s][5]['accum_1s']*100 for s in grand_results[m]]
        m5_val = np.mean(accs_5_arr)
        print(f"{'MEAN':<14} {m5_val:>8.1f} {montage_means[m]['10s']:>8.1f} {montage_means[m]['20s']:>8.1f} {montage_means[m]['40s']:>8.1f}")
        
    # Write summary CSV
    with open(out_dir / "summary.csv", "w") as f:
        f.write("\n".join(summary_csv))
        
    # Write subject detailed results
    sub_csv = ["Montage,Subject,5s,10s,20s,40s"]
    for m in suite_montages:
        for s in grand_results[m]:
            r = grand_results[m][s]
            sub_csv.append(f"{m},{s},{r[5]['accum_1s']*100:.1f},{r[10]['accum_1s']*100:.1f},{r[20]['accum_1s']*100:.1f},{r[40]['accum_1s']*100:.1f}")
    with open(out_dir / "subject_results.csv", "w") as f:
        f.write("\n".join(sub_csv))
        
    # Channel selection frequency
    if "best8_correlation" in suite_montages:
        print("\nCHANNEL SELECTION STABILITY (Data-Driven)")
        print("-" * 50)
        sel_csv = ["Electrode,Selection Frequency"]
        for idx in np.argsort([-channel_selection_freq[i] for i in range(64)]):
            freq = channel_selection_freq[idx]
            if freq > 0:
                print(f"{DTU_CHANNELS[idx]:<12} {freq}/{len(folds)}")
                sel_csv.append(f"{DTU_CHANNELS[idx]},{freq}")
        with open(out_dir / "channel_selection.csv", "w") as f:
            f.write("\n".join(sel_csv))
            
    # Gap Analysis
    if "best8_correlation" in montage_means and "near_ear_strict" in montage_means:
        print("\n========================================================================================")
        print("BEST-8 vs NEAR-EAR-8")
        print("========================================================================================")
        print(f"{'Window':<12} {'Best 8':<12} {'Near-ear 8':<14} {'Gap':<8}")
        print("-" * 50)
        for w in ["10s", "20s", "40s"]:
            best = montage_means["best8_correlation"][w]
            near = montage_means["near_ear_strict"][w]
            gap = best - near
            print(f"{w:<12} {best:<12.1f} {near:<14.1f} {gap:+.1f} pp")
        print("========================================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified 8-Channel Montage Evaluation Runner")
    parser.add_argument("--montage-suite", type=str, choices=["8ch"], help="Run full suite of 8-channel montages")
    parser.add_argument("--montage", type=str, help="Specific montage to run")
    parser.add_argument("--lowcut", type=float, default=1.0)
    parser.add_argument("--highcut", type=float, default=6.0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--subjects", type=str, nargs="+", help="Specific subjects to run (e.g. S1_data_preproc)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()
    
    run_experiment(args)
