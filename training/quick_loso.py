import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy

from preprocessing.dataset import get_mapping_data, subject_files, load_subject_examples
from training.train_matchnet_loso import prepare_dataset, chunk_trial, evaluate_model
from models.matchnet import ContrastiveMatchNet, contrastive_loss

# Hardcode configuration for quick tests
TRAIN_WINDOW_SEC = 2
TRAIN_HOP_SEC = 0.5
EVAL_WINDOW_SEC = 2
FS = 64
CHANNELS = [13, 46, 43, 23, 50, 0, 52, 14]
LOWCUT = 1.0
HIGHCUT = 6.0
NUM_BANDS = 28

def evaluate_model_version(model_lags, X_tr_full, YA_tr_full, YB_tr_full, X_va_full, YA_va_full, YB_va_full, X_te_full, YA_te_full, YB_te_full, device, epochs, batch_size):
    # Chunk training data
    X_tr, YA_tr, YB_tr = [], [], []
    for i in range(len(X_tr_full)):
        cx, cya, cyb = chunk_trial(X_tr_full[i], YA_tr_full[i], YB_tr_full[i], TRAIN_WINDOW_SEC, TRAIN_HOP_SEC)
        X_tr.extend(cx); YA_tr.extend(cya); YB_tr.extend(cyb)
        
    X_tr_t = torch.FloatTensor(np.stack(X_tr))
    YA_tr_t = torch.FloatTensor(np.stack(YA_tr))
    YB_tr_t = torch.FloatTensor(np.stack(YB_tr))
    
    train_dataset = TensorDataset(X_tr_t, YA_tr_t, YB_tr_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        
    model = ContrastiveMatchNet(eeg_model_type="eegnet", eeg_channels=len(CHANNELS), audio_channels=NUM_BANDS, latent_dim=64, lags=model_lags).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
    
    input_dropout = nn.Dropout(p=0.2)
    best_val_acc = 0.0
    best_weights = deepcopy(model.state_dict())
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        for bx, bya, byb in train_loader:
            bx, bya, byb = bx.to(device), bya.to(device), byb.to(device)
            bx, bya, byb = input_dropout(bx), input_dropout(bya), input_dropout(byb)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                z_eeg, z_a, z_b = model(bx, bya, byb)
                loss, sa, sb = contrastive_loss(z_eeg, z_a, z_b, margin=0.05)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        nc_va, nt_va = evaluate_model(model, X_va_full, YA_va_full, YB_va_full, device, window_sec=EVAL_WINDOW_SEC)
        val_acc = nc_va / max(nt_va, 1)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = deepcopy(model.state_dict())
            
    runtime = time.time() - start_time
    
    # Evaluate best test
    model.load_state_dict(best_weights)
    nc_te, nt_te = evaluate_model(model, X_te_full, YA_te_full, YB_te_full, device, window_sec=EVAL_WINDOW_SEC)
    best_test_acc = nc_te / max(nt_te, 1)
    
    return best_val_acc, best_test_acc, runtime

def quick_loso(target_subjects, epochs, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Quick LOSO on device: {device} for {epochs} epochs")
    
    mapping, envelopes = get_mapping_data()
    all_paths = subject_files()
    if not all_paths:
        print("No subjects found.")
        return
        
    subject_examples = {str(p): load_subject_examples(p) for p in all_paths}
    
    results = {}
    
    for held_out_subj in target_subjects:
        held_out_path = next((p for p in all_paths if p.stem == held_out_subj), None)
        if not held_out_path:
            print(f"Warning: Subject {held_out_subj} not found, skipping.")
            continue
            
        print(f"\n--- Testing Subject {held_out_subj} ---")
        
        train_paths = [p for p in all_paths if p.stem != held_out_subj]
        
        # Prepare Data
        train_exs = []
        for p in train_paths: train_exs.extend(subject_examples[str(p)])
        test_exs = subject_examples[str(held_out_path)]
        
        np.random.seed(42)
        np.random.shuffle(train_exs)
        val_split = int(0.1 * len(train_exs))
        val_exs, train_exs = train_exs[:val_split], train_exs[val_split:]
        
        X_tr_full, YA_tr_full, YB_tr_full = [], [], []
        X_va_full, YA_va_full, YB_va_full = [], [], []
        
        for p in train_paths:
            tX, tYA, tYB = prepare_dataset(subject_examples[str(p)], CHANNELS, LOWCUT, HIGHCUT, p.stem, mapping, envelopes)
            v_split_idx = int(0.1 * len(tX))
            X_va_full.extend(tX[:v_split_idx]); YA_va_full.extend(tYA[:v_split_idx]); YB_va_full.extend(tYB[:v_split_idx])
            X_tr_full.extend(tX[v_split_idx:]); YA_tr_full.extend(tYA[v_split_idx:]); YB_tr_full.extend(tYB[v_split_idx:])
            
        X_te_full, YA_te_full, YB_te_full = prepare_dataset(test_exs, CHANNELS, LOWCUT, HIGHCUT, held_out_subj, mapping, envelopes)
        
        print("Running Baseline (No Lag)...")
        b_val, b_test, b_time = evaluate_model_version([], X_tr_full, YA_tr_full, YB_tr_full, X_va_full, YA_va_full, YB_va_full, X_te_full, YA_te_full, YB_te_full, device, epochs, batch_size)
        
        print("Running Lag Model ([3, 6, 10, 13, 16])...")
        l_val, l_test, l_time = evaluate_model_version([3, 6, 10, 13, 16], X_tr_full, YA_tr_full, YB_tr_full, X_va_full, YA_va_full, YB_va_full, X_te_full, YA_te_full, YB_te_full, device, epochs, batch_size)
        
        delta = l_test - b_test
        results[held_out_subj] = {
            "baseline": b_test,
            "lagged": l_test,
            "delta": delta,
            "b_time": b_time,
            "l_time": l_time
        }
        print(f"  Baseline 2s Acc: {b_test*100:.2f}%")
        print(f"  Lagged 2s Acc  : {l_test*100:.2f}%")
        print(f"  Delta          : {delta*100:+.2f}%")
        
    print("\n\n" + "="*50)
    print("SMOKE TEST SUMMARY")
    print("="*50)
    print("| Subject | Baseline | Lag Model | Delta |")
    print("| ------- | -------- | --------- | ----- |")
    deltas = []
    for subj, res in results.items():
        b, l, d = res['baseline']*100, res['lagged']*100, res['delta']*100
        deltas.append(d)
        print(f"| {subj:7s} | {b:7.2f}% | {l:8.2f}% | {d:+4.2f}% |")
        
    if deltas:
        mean_delta = np.mean(deltas)
        median_delta = np.median(deltas)
        print(f"\nMean Delta:   {mean_delta:+.2f}%")
        print(f"Median Delta: {median_delta:+.2f}%")
        
        print("\nDECISION RECOMMENDATION:")
        if mean_delta < 2.0:
            print("❌ Mean gain < 2%. RECOMMENDATION: KILL EXPERIMENT. Do not run full LOSO.")
        elif 2.0 <= mean_delta <= 5.0:
            print("⚠️ Mean gain 2-5%. RECOMMENDATION: Run Mini-LOSO (S1, S4, S6, S8, S11, S14).")
        else:
            print("✅ Mean gain > 5%. RECOMMENDATION: PROMOTE. Run full LOSO immediately.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick LOSO Smoke Test")
    parser.add_argument("--subjects", nargs='+', default=["S8", "S11", "S6"], help="Subjects to evaluate as held-out")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs to train per model")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    args = parser.parse_args()
    
    quick_loso(args.subjects, args.epochs, args.batch_size)
