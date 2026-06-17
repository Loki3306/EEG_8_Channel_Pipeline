import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy
import sys
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from baselines.ridge_aad import load_subject_examples, subject_files
from training.train_matchnet_loso import prepare_dataset, chunk_trial, evaluate_model, get_mapping_data
from models.matchnet import ContrastiveMatchNet, contrastive_loss

# Configuration
TRAIN_WINDOW_SEC = 2.0
TRAIN_HOP_SEC = 0.5
EPOCHS = 20
BATCH_SIZE = 128
LOWCUT = 1.0
HIGHCUT = 6.0
NUM_BANDS = 28
TARGET_SUBJECTS = ["S8", "S9", "S11"]

# Channel index mapping for Biosemi64
BIOSEMI_MAPPING = {
    "Fp1": 0, "AF7": 1, "AF3": 2, "F1": 3, "F3": 4, "F5": 5, "F7": 6, "FT7": 7, "FC5": 8, "FC3": 9,
    "FC1": 10, "C1": 11, "C3": 12, "C5": 13, "T7": 14, "TP7": 15, "CP5": 16, "CP3": 17, "CP1": 18,
    "P1": 19, "P3": 20, "P5": 21, "P7": 22, "P9": 23, "PO7": 24, "PO3": 25, "O1": 26, "Iz": 27,
    "Oz": 28, "POz": 29, "Pz": 30, "CPz": 31, "Fpz": 32, "Fp2": 33, "AF8": 34, "AF4": 35, "AFz": 36,
    "Fz": 37, "F2": 38, "F4": 39, "F6": 40, "F8": 41, "FT8": 42, "FC6": 43, "FC4": 44, "FC2": 45,
    "FCz": 46, "Cz": 47, "C2": 48, "C4": 49, "C6": 50, "T8": 51, "TP8": 52, "CP6": 53, "CP4": 54,
    "CP2": 55, "P2": 56, "P4": 57, "P6": 58, "P8": 59, "P10": 60, "PO8": 61, "PO4": 62, "O2": 63
}

BASELINE_CHANNELS = [13, 46, 43, 23, 50, 0, 52, 14]
SET_B_CHANNELS = [BIOSEMI_MAPPING[ch] for ch in ["T7", "T8", "TP7", "TP8", "Cz", "FCz", "C5", "C6"]]

def apply_delta(X_list):
    """
    Augments each EEG trial with its temporal derivative.
    Input shape: [Channels, T]
    Output shape: [2 * Channels, T]
    """
    augmented = []
    for x in X_list:
        delta = np.diff(x, axis=-1, prepend=x[..., :1])
        x_aug = np.concatenate([x, delta], axis=0)
        augmented.append(x_aug)
    return augmented

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_and_eval(num_channels, X_tr_full, YA_tr_full, YB_tr_full, X_va_full, YA_va_full, YB_va_full, X_te_full, YA_te_full, YB_te_full, device):
    # Chunk training data
    X_tr, YA_tr, YB_tr = [], [], []
    for i in range(len(X_tr_full)):
        cx, cya, cyb = chunk_trial(X_tr_full[i], YA_tr_full[i], YB_tr_full[i], TRAIN_WINDOW_SEC, TRAIN_HOP_SEC)
        X_tr.extend(cx); YA_tr.extend(cya); YB_tr.extend(cyb)
        
    X_tr_t = torch.FloatTensor(np.stack(X_tr))
    YA_tr_t = torch.FloatTensor(np.stack(YA_tr))
    YB_tr_t = torch.FloatTensor(np.stack(YB_tr))
    
    train_dataset = TensorDataset(X_tr_t, YA_tr_t, YB_tr_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
        
    model = ContrastiveMatchNet(eeg_model_type="eegnet", eeg_channels=num_channels, audio_channels=NUM_BANDS, latent_dim=64, lags=[], audio_model_type="standard", temporal_pooling=False).to(device)
    
    print(f"    -> Initialized Model with {num_channels} EEG input channels.")
    print(f"    -> Parameter Count: {count_parameters(model):,}")
    
    # Just to verify shape on first batch
    for bx, _, _ in train_loader:
        print(f"    -> Input Tensor Shape: {bx.shape}")
        break
        
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
    
    input_dropout = nn.Dropout(p=0.2)
    best_val_acc = 0.0
    best_weights = deepcopy(model.state_dict())
    
    for epoch in range(EPOCHS):
        model.train()
        for bx, bya, byb in train_loader:
            bx, bya, byb = bx.to(device), bya.to(device), byb.to(device)
            bx, bya, byb = input_dropout(bx), input_dropout(bya), input_dropout(byb)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                z_eeg, z_a, z_b = model(bx, bya, byb)
                loss, sa, sb = contrastive_loss(z_eeg, z_a, z_b, margin=0.05, model=model)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        nc_va, nt_va = evaluate_model(model, X_va_full, YA_va_full, YB_va_full, device, window_sec=2.0)
        val_acc = nc_va / max(nt_va, 1)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = deepcopy(model.state_dict())
            
    # Load best model for evaluation
    model.load_state_dict(best_weights)
    
    # Evaluate 2s
    nc_2s, nt_2s = evaluate_model(model, X_te_full, YA_te_full, YB_te_full, device, window_sec=2.0)
    acc_2s = nc_2s / max(nt_2s, 1)
    
    # Evaluate 5s
    nc_5s, nt_5s = evaluate_model(model, X_te_full, YA_te_full, YB_te_full, device, window_sec=5.0)
    acc_5s = nc_5s / max(nt_5s, 1)
    
    return acc_2s, acc_5s

def prepare_data_for_variant(subject_examples, train_paths, test_path, channels, mapping, envelopes, use_delta):
    X_tr_full, YA_tr_full, YB_tr_full = [], [], []
    X_va_full, YA_va_full, YB_va_full = [], [], []
    
    for p in train_paths:
        tX, tYA, tYB = prepare_dataset(subject_examples[str(p)], channels, LOWCUT, HIGHCUT, p.stem, mapping, envelopes)
        if use_delta:
            tX = apply_delta(tX)
            
        v_split_idx = int(0.1 * len(tX))
        X_va_full.extend(tX[:v_split_idx]); YA_va_full.extend(tYA[:v_split_idx]); YB_va_full.extend(tYB[:v_split_idx])
        X_tr_full.extend(tX[v_split_idx:]); YA_tr_full.extend(tYA[v_split_idx:]); YB_tr_full.extend(tYB[v_split_idx:])
        
    X_te_full, YA_te_full, YB_te_full = prepare_dataset(subject_examples[str(test_path)], channels, LOWCUT, HIGHCUT, test_path.stem, mapping, envelopes)
    if use_delta:
        X_te_full = apply_delta(X_te_full)
        
    return X_tr_full, YA_tr_full, YB_tr_full, X_va_full, YA_va_full, YB_va_full, X_te_full, YA_te_full, YB_te_full

def run_study():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Delta-EEG Channel Study on device: {device}")
    print("="*60)
    
    mapping, envelopes = get_mapping_data()
    all_paths = subject_files()
    if not all_paths:
        print("No subjects found.")
        return
        
    subject_examples = {str(p): load_subject_examples(p) for p in all_paths}
    
    variants = ["Baseline", "DeltaEEG", "DeltaEEG+SetB"]
    results = {subj: {v: {} for v in variants} for subj in TARGET_SUBJECTS}
    
    for held_out_subj in TARGET_SUBJECTS:
        held_out_path = next((p for p in all_paths if p.stem.split('_')[0] == held_out_subj), None)
        if not held_out_path:
            continue
            
        print(f"\n" + "-"*40)
        print(f"Processing Subject {held_out_subj}")
        print("-"*40)
        
        train_paths = [p for p in all_paths if p.stem.split('_')[0] != held_out_subj]
        
        # 1. Baseline
        print(f"\n[ Variant 1: Baseline ]")
        data = prepare_data_for_variant(subject_examples, train_paths, held_out_path, BASELINE_CHANNELS, mapping, envelopes, use_delta=False)
        acc_2s, acc_5s = train_and_eval(len(BASELINE_CHANNELS), *data, device)
        results[held_out_subj]["Baseline"]["2s"] = acc_2s
        results[held_out_subj]["Baseline"]["5s"] = acc_5s
        
        # 2. Delta EEG
        print(f"\n[ Variant 2: DeltaEEG ]")
        data = prepare_data_for_variant(subject_examples, train_paths, held_out_path, BASELINE_CHANNELS, mapping, envelopes, use_delta=True)
        acc_2s, acc_5s = train_and_eval(len(BASELINE_CHANNELS) * 2, *data, device)
        results[held_out_subj]["DeltaEEG"]["2s"] = acc_2s
        results[held_out_subj]["DeltaEEG"]["5s"] = acc_5s
        
        # 3. Delta EEG + Set B
        print(f"\n[ Variant 3: DeltaEEG+SetB ]")
        data = prepare_data_for_variant(subject_examples, train_paths, held_out_path, SET_B_CHANNELS, mapping, envelopes, use_delta=True)
        acc_2s, acc_5s = train_and_eval(len(SET_B_CHANNELS) * 2, *data, device)
        results[held_out_subj]["DeltaEEG+SetB"]["2s"] = acc_2s
        results[held_out_subj]["DeltaEEG+SetB"]["5s"] = acc_5s

    # Print Output Tables
    print("\n\n" + "="*80)
    print("PER SUBJECT ACCURACY (2s Window)")
    print("="*80)
    print("| Subject | Baseline | DeltaEEG | DeltaEEG+SetB |")
    print("| ------- | -------- | -------- | ------------- |")
    for subj in TARGET_SUBJECTS:
        if subj not in results or "Baseline" not in results[subj]: continue
        r = results[subj]
        print(f"| {subj:7s} | {r['Baseline']['2s']*100:8.2f} | {r['DeltaEEG']['2s']*100:8.2f} | {r['DeltaEEG+SetB']['2s']*100:13.2f} |")

    print("\n" + "="*80)
    print("PER SUBJECT ACCURACY (5s Window)")
    print("="*80)
    print("| Subject | Baseline | DeltaEEG | DeltaEEG+SetB |")
    print("| ------- | -------- | -------- | ------------- |")
    for subj in TARGET_SUBJECTS:
        if subj not in results or "Baseline" not in results[subj]: continue
        r = results[subj]
        print(f"| {subj:7s} | {r['Baseline']['5s']*100:8.2f} | {r['DeltaEEG']['5s']*100:8.2f} | {r['DeltaEEG+SetB']['5s']*100:13.2f} |")

    print("\n" + "="*80)
    print("MEAN ACCURACY AND GAINS")
    print("="*80)
    
    means_2s = {}
    means_5s = {}
    
    for v in variants:
        means_2s[v] = np.mean([results[s][v]["2s"] for s in TARGET_SUBJECTS if v in results[s]]) * 100
        means_5s[v] = np.mean([results[s][v]["5s"] for s in TARGET_SUBJECTS if v in results[s]]) * 100
        
    print("For 2s:")
    print("| Variant       | Mean Accuracy | Delta vs Baseline |")
    print("| ------------- | ------------- | ----------------- |")
    for v in variants:
        gain = means_2s[v] - means_2s["Baseline"]
        print(f"| {v:13s} | {means_2s[v]:13.2f} | {gain:+17.2f}% |")

    print("\nFor 5s:")
    print("| Variant       | Mean Accuracy | Delta vs Baseline |")
    print("| ------------- | ------------- | ----------------- |")
    for v in variants:
        gain = means_5s[v] - means_5s["Baseline"]
        print(f"| {v:13s} | {means_5s[v]:13.2f} | {gain:+17.2f}% |")
        
    print("\n" + "="*80)
    print("DECISION LOGIC (Average Gain across 2s & 5s)")
    print("="*80)
    
    base_2s = means_2s["Baseline"]
    base_5s = means_5s["Baseline"]
    
    for v in ["DeltaEEG", "DeltaEEG+SetB"]:
        gain_2s = means_2s[v] - base_2s
        gain_5s = means_5s[v] - base_5s
        avg_gain = (gain_2s + gain_5s) / 2.0
        
        if avg_gain < 1.0:
            rec = "KILL"
        elif avg_gain <= 2.0:
            rec = "Mini-LOSO"
        else:
            rec = "Promote to Full LOSO"
        
        print(f"{v}: {avg_gain:+.2f}% -> {rec}")

if __name__ == "__main__":
    run_study()
