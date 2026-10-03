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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.catcn import CATCNDirectDecoder
from baselines.ridge_aad import load_subject_examples, subject_files, iter_leave_one_subject_out
from training.train_matchnet_wavlm import (
    FS, TRAIN_WINDOW_SEC, TRAIN_HOP_SEC,
    prepare_dataset, select_top_channels_from_train, get_mapping_data, ChunkDataset
)

def evaluate_catcn_multiwindow(model, X, Y_A, Y_B, device):
    """
    Evaluates CA-TCN across multi-window decision lengths [1s, 2s, 5s, 10s, 15s, 20s, 25s, 30s, 35s, 40s].
    Uses direct classification logit accumulation: D = sum_t (logit_A(t) - logit_B(t)).
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
        # 1. 1s-accumulated direct logits & majority vote
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
        
        # 2. 5s-accumulated direct logits
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
            
        # 3. Direct independent window classification
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

def train_catcn_loso(channels=None, num_channels=64, rank_channels=False, lowcut=1.0, highcut=6.0, batch_size=128, num_workers=0, subjects_to_run=None, epochs=30, lr=1e-3, checkpoint_dir="checkpoints_catcn"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_paths = subject_files()
    if not all_paths:
        print("No subjects found.")
        return
        
    subject_examples = {str(p): load_subject_examples(p) for p in all_paths}
    folds = list(iter_leave_one_subject_out(all_paths))
    if subjects_to_run:
        folds = [f for f in folds if f[0].stem in subjects_to_run]
        
    os.makedirs(checkpoint_dir, exist_ok=True)
    grand_summary = {}
    
    print("\n" + "="*70)
    print(f" CA-TCN DIRECT AAD CLASSIFIER (2026 ARCHITECTURE)")
    print(f" EEG Channels: {num_channels} | Audio: Gammatone 28-band | Device: {device}")
    print(f" Folds to evaluate: {len(folds)} subject(s)")
    print("="*70 + "\n")
    
    for held_out_path, train_paths in folds:
        sub_name = held_out_path.stem
        print(f"\n{'='*60}")
        print(f" FOLD: Held-out Subject {sub_name}")
        print(f" Pre-fold RAM: {psutil.virtual_memory().percent}% ({psutil.virtual_memory().used / 1e9:.2f} GB used)")
        print(f"{'='*60}")
        
        mapping, envelopes = get_mapping_data("gammatone")
        
        # Channel Selection
        if channels is not None and len(channels) > 0:
            fold_channels = list(channels)
            print(f"  [Channel Setup]: Using explicit user channels ({len(fold_channels)} ch): {fold_channels}")
        elif rank_channels and num_channels < 64:
            print(f"  [Channel Setup]: Ranking channels on TRAINING subjects only ({len(train_paths)} subjects, 0% test leakage)...")
            fold_channels, _ = select_top_channels_from_train(
                subject_examples, train_paths, mapping, envelopes, num_channels, lowcut, highcut
            )
            print(f"  [Channel Setup]: Selected Top {num_channels} channels: {fold_channels}")
        else:
            fold_channels = list(range(num_channels))
            print(f"  [Channel Setup]: Using standard {num_channels} channels [0 to {num_channels-1}]")
            
        test_exs = subject_examples[str(held_out_path)]
        X_va_full, YA_va_full, YB_va_full = [], [], []
        X_tr_full, YA_tr_full, YB_tr_full = [], [], []
        
        for p in train_paths:
            exs = list(subject_examples[str(p)])
            np.random.seed(42)
            np.random.shuffle(exs)
            val_split_num = max(1, int(0.1 * len(exs))) if len(exs) > 0 else 0
            val_exs = exs[:val_split_num]
            train_exs = exs[val_split_num:]
            
            tX, tYA, tYB = prepare_dataset(train_exs, fold_channels, lowcut, highcut, p.stem, mapping, envelopes)
            X_tr_full.extend(tX); YA_tr_full.extend(tYA); YB_tr_full.extend(tYB)
            
            vX, vYA, vYB = prepare_dataset(val_exs, fold_channels, lowcut, highcut, p.stem, mapping, envelopes)
            X_va_full.extend(vX); YA_va_full.extend(vYA); YB_va_full.extend(vYB)
            
        X_te_full, YA_te_full, YB_te_full = prepare_dataset(test_exs, fold_channels, lowcut, highcut, sub_name, mapping, envelopes)
        del envelopes
        gc.collect()
        
        # Convert to tensor lists
        X_tr_full = [torch.from_numpy(x) for x in X_tr_full]
        YA_tr_full = [torch.from_numpy(x) for x in YA_tr_full]
        YB_tr_full = [torch.from_numpy(x) for x in YB_tr_full]
        
        # 75% overlap for training chunks (arXiv:2603.26394 Sec 2.7)
        chunk_indices = []
        win_samples = int(TRAIN_WINDOW_SEC * FS)
        hop_samples = int(1.25 * FS)
        for i in range(len(X_tr_full)):
            trial_len = X_tr_full[i].shape[1]
            start = 0
            while start + win_samples <= trial_len:
                chunk_indices.append((i, start, start + win_samples))
                start += hop_samples
                
        train_dataset = ChunkDataset(X_tr_full, YA_tr_full, YB_tr_full, chunk_indices)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        
        # Prepare validation chunks for continuous, low-variance validation loss
        val_chunk_indices = []
        for i in range(len(X_va_full)):
            trial_len = X_va_full[i].shape[1]
            start = 0
            while start + win_samples <= trial_len:
                val_chunk_indices.append((i, start, start + win_samples))
                start += hop_samples
        val_dataset = ChunkDataset(X_va_full, YA_va_full, YB_va_full, val_chunk_indices)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        model = CATCNDirectDecoder(
            eeg_channels=len(fold_channels),
            audio_channels=1,
            hidden_dim=64,
            max_lag_samples=8,
            dropout=0.2
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
        
        best_val_loss = float('inf')
        best_weights = deepcopy(model.state_dict())
        min_epochs = 15
        patience = 12
        epochs_no_improve = 0
        
        print(f"Training CA-TCN on {len(chunk_indices)} chunks ({TRAIN_WINDOW_SEC}s, 75% overlap) | Batch Size: {batch_size} | LR: {lr}...")
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            correct_train = 0.0
            total_train = 0
            
            for bx, bya, byb in train_loader:
                bx = bx.to(device, non_blocking=True)
                bya = bya.to(device, non_blocking=True)
                byb = byb.to(device, non_blocking=True)
                
                # Prevent presentation order bias (arXiv:2603.26394 Sec 2.7)
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
                    delta, (la, lb), _ = model(bx, c1, c2)
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
            train_epoch_acc = (correct_train / total_train) * 100.0
            
            # Continuous validation evaluation
            model.eval()
            val_loss = 0.0
            val_correct = 0.0
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
                    val_correct += ((delta > 0).float() == target).sum().item()
                    val_total += bx.size(0)
                    
            epoch_val_loss = val_loss / max(val_total, 1)
            epoch_val_acc = (val_correct / max(val_total, 1)) * 100.0
            
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_weights = deepcopy(model.state_dict())
                epochs_no_improve = 0
            elif epoch >= min_epochs:
                epochs_no_improve += 1
                
            print(f"  Epoch {epoch+1:02d}/{epochs} | Train Loss: {train_epoch_loss:.4f} | Train Acc: {train_epoch_acc:.2f}% | Val Loss: {epoch_val_loss:.4f} | Val Acc (5s): {epoch_val_acc:.2f}% | Patience: {epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:
                print(f"  [Early Stopping Triggered at Epoch {epoch+1}] Best Val Loss: {best_val_loss:.4f}")
                break
                
        best_path = Path(checkpoint_dir) / f"catcn_fold_{sub_name}_best.pth"
        torch.save(best_weights, best_path)
        model.load_state_dict(best_weights)
        
        del X_tr_full, YA_tr_full, YB_tr_full, X_va_full, YA_va_full, YB_va_full, train_dataset, train_loader
        gc.collect()
        
        # Test Evaluation across all decision windows
        print(f"\n  [EVALUATION: CA-TCN Multi-Window Breakdown for {sub_name}]")
        multi_win = evaluate_catcn_multiwindow(model, X_te_full, YA_te_full, YB_te_full, device)
        
        print("\n" + "="*88)
        print(f" CA-TCN DIRECT AAD SUMMARY - HELD-OUT: {sub_name} ({len(fold_channels)} Channels)")
        print("="*88)
        print(f" {'Window':>6} | {'1s-Accum Logits':>15} | {'1s-Majority':>12} | {'5s-Accum Logits':>15} | {'Direct Eval':>12} | {'Decisions':>10}")
        print("-" * 88)
        for w in [1, 2, 5, 10, 15, 20, 25, 30, 35, 40]:
            r = multi_win[w]
            s_5s = f"{r['accum_5s']*100:13.2f}%" if r['accum_5s'] is not None else "          N/A"
            print(f" {w:4d} s | {r['accum_1s']*100:13.2f}% | {r['vote_1s']*100:10.2f}% | {s_5s} | {r['direct']*100:10.2f}% | {r['decisions']:10d}")
        print("="*88)
        
        grand_summary[sub_name] = multi_win
        with open(Path(checkpoint_dir) / f"catcn_fold_{sub_name}_metrics.json", "w") as f:
            json.dump(multi_win, f, indent=4)
            
        del X_te_full, YA_te_full, YB_te_full
        gc.collect()

    if len(grand_summary) > 1 or len(folds) == 18:
        print("\n" + "#"*92)
        print(f" FULL DTU CA-TCN LEAVE-ONE-SUBJECT-OUT (LOSO) GRAND BENCHMARK ({num_channels} CHANNELS)")
        print("#"*92)
        print(f" {'Subject':>10} | {'5s (Accum)':>12} | {'10s (Accum)':>12} | {'15s (Accum)':>12} | {'20s (Accum)':>12} | {'30s (Accum)':>12} | {'40s (Accum)':>12}")
        print("-" * 92)
        win_keys = [5, 10, 15, 20, 30, 40]
        col_accs = {w: [] for w in win_keys}
        for sub, wins in grand_summary.items():
            row_str = f" {sub:>10} |"
            for w in win_keys:
                acc = wins[w]["accum_1s"] * 100
                col_accs[w].append(acc)
                row_str += f" {acc:10.2f}% |"
            print(row_str)
        print("-" * 92)
        mean_row = f" {'MEAN':>10} |"
        std_row = f" {'STD':>10} |"
        for w in win_keys:
            mean_row += f" {np.mean(col_accs[w]):10.2f}% |"
            std_row += f" {np.std(col_accs[w]):10.2f}% |"
        print(mean_row)
        print(std_row)
        print("#"*92)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate CA-TCN Direct AAD Classifier")
    parser.add_argument("--channels", type=int, nargs='+', default=None, help="Explicit channel indices")
    parser.add_argument("--num_channels", type=int, default=64, choices=[8, 16, 32, 64], help="Channel count")
    parser.add_argument("--rank_channels", action="store_true", help="Rank channels using only training subjects")
    parser.add_argument("--lowcut", type=float, default=1.0)
    parser.add_argument("--highcut", type=float, default=6.0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--subjects", type=str, nargs="+", help="Specific subjects to run (e.g. S1_data_preproc)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_catcn")
    args = parser.parse_args()
    
    train_catcn_loso(
        channels=args.channels,
        num_channels=args.num_channels,
        rank_channels=args.rank_channels,
        lowcut=args.lowcut,
        highcut=args.highcut,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        subjects_to_run=args.subjects,
        epochs=args.epochs,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir
    )
