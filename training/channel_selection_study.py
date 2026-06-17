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

# Channel index mapping for Biosemi64 (verified via mne)
BIOSEMI_MAPPING = {
    "Fp1": 0, "AF7": 1, "AF3": 2, "F1": 3, "F3": 4, "F5": 5, "F7": 6, "FT7": 7, "FC5": 8, "FC3": 9,
    "FC1": 10, "C1": 11, "C3": 12, "C5": 13, "T7": 14, "TP7": 15, "CP5": 16, "CP3": 17, "CP1": 18,
    "P1": 19, "P3": 20, "P5": 21, "P7": 22, "P9": 23, "PO7": 24, "PO3": 25, "O1": 26, "Iz": 27,
    "Oz": 28, "POz": 29, "Pz": 30, "CPz": 31, "Fpz": 32, "Fp2": 33, "AF8": 34, "AF4": 35, "AFz": 36,
    "Fz": 37, "F2": 38, "F4": 39, "F6": 40, "F8": 41, "FT8": 42, "FC6": 43, "FC4": 44, "FC2": 45,
    "FCz": 46, "Cz": 47, "C2": 48, "C4": 49, "C6": 50, "T8": 51, "TP8": 52, "CP6": 53, "CP4": 54,
    "CP2": 55, "P2": 56, "P4": 57, "P6": 58, "P8": 59, "P10": 60, "PO8": 61, "PO4": 62, "O2": 63
}

# Define Channel Sets
CHANNEL_SETS = {
    "Baseline": [13, 46, 43, 23, 50, 0, 52, 14], # C5, FCz, FC6, P9, C6, Fp1, TP8, T7
    "Set A": [BIOSEMI_MAPPING[ch] for ch in ["T7", "T8", "TP7", "TP8", "C5", "C6", "FC5", "FC6"]],
    "Set B": [BIOSEMI_MAPPING[ch] for ch in ["T7", "T8", "TP7", "TP8", "Cz", "FCz", "C5", "C6"]],
    "Set C": [BIOSEMI_MAPPING[ch] for ch in ["T7", "T8", "TP7", "TP8", "FT7", "FT8", "C5", "C6"]]
}

def print_mapping_verification():
    print("="*40)
    print("CHANNEL MAPPING VERIFICATION")
    print("="*40)
    
    # Reverse mapping for display
    inv_map = {v: k for k, v in BIOSEMI_MAPPING.items()}
    
    for set_name, indices in CHANNEL_SETS.items():
        names = [inv_map[i] for i in indices]
        print(f"\n{set_name}:")
        print(f"Names:   {names}")
        print(f"Indices: {indices}")
    print("\n" + "="*40 + "\n")


def train_and_eval(channels, X_tr_full, YA_tr_full, YB_tr_full, X_va_full, YA_va_full, YB_va_full, X_te_full, YA_te_full, YB_te_full, device):
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
        
    model = ContrastiveMatchNet(eeg_model_type="eegnet", eeg_channels=len(channels), audio_channels=NUM_BANDS, latent_dim=64, lags=[], audio_model_type="standard", temporal_pooling=False).to(device)
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

def run_study():
    print_mapping_verification()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Study on device: {device}")
    
    mapping, envelopes = get_mapping_data()
    all_paths = subject_files()
    if not all_paths:
        print("No subjects found.")
        return
        
    subject_examples = {str(p): load_subject_examples(p) for p in all_paths}
    
    # Structure: results[subject][set_name][window] = accuracy
    results = {subj: {s_name: {} for s_name in CHANNEL_SETS.keys()} for subj in TARGET_SUBJECTS}
    
    for held_out_subj in TARGET_SUBJECTS:
        held_out_path = next((p for p in all_paths if p.stem.split('_')[0] == held_out_subj), None)
        if not held_out_path:
            print(f"Warning: Subject {held_out_subj} not found, skipping.")
            continue
            
        print(f"\n--- Processing Subject {held_out_subj} ---")
        
        train_paths = [p for p in all_paths if p.stem.split('_')[0] != held_out_subj]
        test_exs = subject_examples[str(held_out_path)]
        
        for set_name, channels in CHANNEL_SETS.items():
            print(f"  Training {set_name}...")
            
            X_tr_full, YA_tr_full, YB_tr_full = [], [], []
            X_va_full, YA_va_full, YB_va_full = [], [], []
            
            for p in train_paths:
                tX, tYA, tYB = prepare_dataset(subject_examples[str(p)], channels, LOWCUT, HIGHCUT, p.stem, mapping, envelopes)
                v_split_idx = int(0.1 * len(tX))
                X_va_full.extend(tX[:v_split_idx]); YA_va_full.extend(tYA[:v_split_idx]); YB_va_full.extend(tYB[:v_split_idx])
                X_tr_full.extend(tX[v_split_idx:]); YA_tr_full.extend(tYA[v_split_idx:]); YB_tr_full.extend(tYB[v_split_idx:])
                
            X_te_full, YA_te_full, YB_te_full = prepare_dataset(test_exs, channels, LOWCUT, HIGHCUT, held_out_path.stem, mapping, envelopes)
            
            acc_2s, acc_5s = train_and_eval(channels, X_tr_full, YA_tr_full, YB_tr_full, X_va_full, YA_va_full, YB_va_full, X_te_full, YA_te_full, YB_te_full, device)
            
            results[held_out_subj][set_name]["2s"] = acc_2s
            results[held_out_subj][set_name]["5s"] = acc_5s
            
            print(f"    2s Accuracy: {acc_2s*100:.2f}% | 5s Accuracy: {acc_5s*100:.2f}%")

    print("\n\n" + "="*60)
    print("PER SUBJECT RESULTS")
    print("="*60)
    print("2s Accuracy:")
    print("| Subject | Baseline | Set A    | Set B    | Set C    |")
    print("| ------- | -------- | -------- | -------- | -------- |")
    for subj in TARGET_SUBJECTS:
        if subj not in results or "Baseline" not in results[subj]: continue
        res = results[subj]
        print(f"| {subj:7s} | {res['Baseline']['2s']*100:8.2f} | {res['Set A']['2s']*100:8.2f} | {res['Set B']['2s']*100:8.2f} | {res['Set C']['2s']*100:8.2f} |")

    print("\n5s Accuracy:")
    print("| Subject | Baseline | Set A    | Set B    | Set C    |")
    print("| ------- | -------- | -------- | -------- | -------- |")
    for subj in TARGET_SUBJECTS:
        if subj not in results or "Baseline" not in results[subj]: continue
        res = results[subj]
        print(f"| {subj:7s} | {res['Baseline']['5s']*100:8.2f} | {res['Set A']['5s']*100:8.2f} | {res['Set B']['5s']*100:8.2f} | {res['Set C']['5s']*100:8.2f} |")

    print("\n" + "="*60)
    print("MEAN PERFORMANCE")
    print("="*60)
    
    means_2s, means_5s = {}, {}
    for set_name in CHANNEL_SETS.keys():
        m2 = np.mean([results[s][set_name]["2s"] for s in TARGET_SUBJECTS if set_name in results[s]]) * 100
        m5 = np.mean([results[s][set_name]["5s"] for s in TARGET_SUBJECTS if set_name in results[s]]) * 100
        means_2s[set_name] = m2
        means_5s[set_name] = m5
        
    print("For 2s:")
    print("| Set      | Mean Accuracy |")
    print("| -------- | ------------- |")
    for set_name in CHANNEL_SETS.keys():
        print(f"| {set_name:8s} | {means_2s[set_name]:13.2f} |")

    print("\nFor 5s:")
    print("| Set      | Mean Accuracy |")
    print("| -------- | ------------- |")
    for set_name in CHANNEL_SETS.keys():
        print(f"| {set_name:8s} | {means_5s[set_name]:13.2f} |")

    print("\n" + "="*60)
    print("DELTA VS CURRENT CHANNELS")
    print("="*60)
    
    for w, means in zip(["2s", "5s"], [means_2s, means_5s]):
        base = means["Baseline"]
        print(f"\nFor {w}:")
        for set_name in ["Set A", "Set B", "Set C"]:
            delta = means[set_name] - base
            print(f"{set_name} - Baseline : {delta:+.2f}%")

    print("\n" + "="*60)
    print("DECISION LOGIC")
    print("="*60)
    
    print("\nRecommendations based on Average Mean Gain (2s + 5s) vs Baseline:")
    base_2s = means_2s["Baseline"]
    base_5s = means_5s["Baseline"]
    
    for set_name in ["Set A", "Set B", "Set C"]:
        gain_2s = means_2s[set_name] - base_2s
        gain_5s = means_5s[set_name] - base_5s
        avg_gain = (gain_2s + gain_5s) / 2.0
        
        if avg_gain < 1.0:
            rec = "KILL"
        elif avg_gain <= 2.0:
            rec = "CONSIDER"
        else:
            rec = "PROMOTE"
        print(f"{set_name}: {avg_gain:+.2f}% (2s: {gain_2s:+.2f}%, 5s: {gain_5s:+.2f}%) -> {rec}")

if __name__ == "__main__":
    run_study()
