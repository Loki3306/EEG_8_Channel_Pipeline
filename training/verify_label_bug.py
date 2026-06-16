import argparse
import sys
import os
import json
import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
from copy import deepcopy
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.matchnet import ContrastiveMatchNet, contrastive_loss
from baselines.ridge_aad import load_subject_examples, subject_files, iter_leave_one_subject_out
from training.train_matchnet_loso import get_mapping_data, chunk_trial, evaluate_model, normalize_array
from scipy.signal import butter, filtfilt
from torch.utils.data import TensorDataset, DataLoader

FS = 64
CHANNELS = [13, 46, 43, 23, 50, 0, 52, 14]
LOWCUT = 1.0
HIGHCUT = 6.0
NUM_BANDS = 1 # Gammatone
TRAIN_WINDOW_SEC = 2.0
TRAIN_HOP_SEC = 1.0
EVAL_WINDOW_SEC = 10.0

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4, axis=-1):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data, axis=axis)

# --- OLD BUGGY PREPARE ---
def prepare_dataset_old(examples, channels, lowcut, highcut, subject_id, mapping, envelopes):
    X, Y_A, Y_B = [], [], []
    sub_key = subject_id.replace("_data_preproc", "")
    for i, ex in enumerate(examples):
        eeg = butter_bandpass_filter(ex.eeg[:, channels].T, lowcut, highcut, FS, axis=1)
        x_norm = normalize_array(eeg.T).T 
        trial_key = f"trial_{i}"
        
        if sub_key in mapping and trial_key in mapping[sub_key]:
            fname_a = mapping[sub_key][trial_key]["wavA"]["filename"]
            fname_b = mapping[sub_key][trial_key]["wavB"]["filename"]
            env_a_full, env_b_full = envelopes[fname_a], envelopes[fname_b]
        else:
            continue
            
        min_len = min(x_norm.shape[1], env_a_full.shape[1])
        x_norm = x_norm[:, :min_len]
        env_a = normalize_array(env_a_full[:, :min_len].T).T
        env_b = normalize_array(env_b_full[:, :min_len].T).T
        
        X.append(x_norm)
        Y_A.append(env_a) # BUG: ALWAYS WAV A
        Y_B.append(env_b)
    return X, Y_A, Y_B

# --- NEW PATCHED PREPARE ---
def prepare_dataset_new(examples, channels, lowcut, highcut, subject_id, mapping, envelopes):
    X, Y_A, Y_B = [], [], []
    sub_key = subject_id.replace("_data_preproc", "")
    for i, ex in enumerate(examples):
        eeg = butter_bandpass_filter(ex.eeg[:, channels].T, lowcut, highcut, FS, axis=1)
        x_norm = normalize_array(eeg.T).T 
        trial_key = f"trial_{i}"
        
        if sub_key in mapping and trial_key in mapping[sub_key]:
            fname_a = mapping[sub_key][trial_key]["wavA"]["filename"]
            fname_b = mapping[sub_key][trial_key]["wavB"]["filename"]
            env_a_full, env_b_full = envelopes[fname_a], envelopes[fname_b]
        else:
            continue
            
        min_len = min(x_norm.shape[1], env_a_full.shape[1])
        x_norm = x_norm[:, :min_len]
        env_a = normalize_array(env_a_full[:, :min_len].T).T
        env_b = normalize_array(env_b_full[:, :min_len].T).T
        
        X.append(x_norm)
        if ex.label == 1:
            Y_A.append(env_a)
            Y_B.append(env_b)
        else:
            Y_A.append(env_b)
            Y_B.append(env_a)
    return X, Y_A, Y_B

def run_experiment(target_subject, use_patched_loader, epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mapping, envelopes = get_mapping_data()
    all_paths = subject_files()
    subject_examples = {str(p): load_subject_examples(p) for p in all_paths}
    
    held_out_path = next((p for p in all_paths if target_subject in p.stem), None)
    if not held_out_path:
        raise ValueError(f"Subject {target_subject} not found")
        
    train_paths = [p for p in all_paths if p != held_out_path]
    
    X_tr_full, YA_tr_full, YB_tr_full = [], [], []
    X_va_full, YA_va_full, YB_va_full = [], [], []
    
    prepare_fn = prepare_dataset_new if use_patched_loader else prepare_dataset_old
    
    print(f"Loading data... (Patched: {use_patched_loader})")
    for p in train_paths:
        tX, tYA, tYB = prepare_fn(subject_examples[str(p)], CHANNELS, LOWCUT, HIGHCUT, p.stem, mapping, envelopes)
        v_split_idx = int(0.1 * len(tX))
        X_va_full.extend(tX[:v_split_idx])
        YA_va_full.extend(tYA[:v_split_idx])
        YB_va_full.extend(tYB[:v_split_idx])
        X_tr_full.extend(tX[v_split_idx:])
        YA_tr_full.extend(tYA[v_split_idx:])
        YB_tr_full.extend(tYB[v_split_idx:])
        
    X_te_full, YA_te_full, YB_te_full = prepare_fn(subject_examples[str(held_out_path)], CHANNELS, LOWCUT, HIGHCUT, held_out_path.stem, mapping, envelopes)
    
    X_tr, YA_tr, YB_tr = [], [], []
    for i in range(len(X_tr_full)):
        cx, cya, cyb = chunk_trial(X_tr_full[i], YA_tr_full[i], YB_tr_full[i], TRAIN_WINDOW_SEC, TRAIN_HOP_SEC)
        X_tr.extend(cx); YA_tr.extend(cya); YB_tr.extend(cyb)
        
    train_dataset = TensorDataset(torch.FloatTensor(np.stack(X_tr)), torch.FloatTensor(np.stack(YA_tr)), torch.FloatTensor(np.stack(YB_tr)))
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True)
    
    model = ContrastiveMatchNet(eeg_model_type='eegnet', eeg_channels=len(CHANNELS), audio_channels=NUM_BANDS, latent_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    val_acc_history = []
    test_acc_history = []
    
    print(f"Starting Training for {target_subject}...")
    for epoch in range(epochs):
        model.train()
        for bx, bya, byb in train_loader:
            bx, bya, byb = bx.to(device), bya.to(device), byb.to(device)
            optimizer.zero_grad()
            
            if scaler:
                with torch.amp.autocast('cuda'):
                    z_eeg, z_a, z_b = model(bx, bya, byb)
                    loss, _, _ = contrastive_loss(z_eeg, z_a, z_b, margin=0.1)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                z_eeg, z_a, z_b = model(bx, bya, byb)
                loss, _, _ = contrastive_loss(z_eeg, z_a, z_b, margin=0.1)
                loss.backward()
                optimizer.step()
                
        # Evaluate Validation
        nc_va, nt_va = evaluate_model(model, X_va_full, YA_va_full, YB_va_full, device, window_sec=EVAL_WINDOW_SEC)
        val_acc = nc_va / max(nt_va, 1)
        val_acc_history.append(val_acc)
        
        # Evaluate Test
        nc_te, nt_te = evaluate_model(model, X_te_full, YA_te_full, YB_te_full, device, window_sec=EVAL_WINDOW_SEC)
        test_acc = nc_te / max(nt_te, 1)
        test_acc_history.append(test_acc)
        
        print(f"Epoch {epoch+1:02d} | Val Acc: {val_acc:.3f} | Test Acc: {test_acc:.3f}")
        
    return val_acc_history, test_acc_history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True, help="Subject to hold out (e.g., S8, S11)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train")
    args = parser.parse_args()
    
    print(f"=== RUNNING EXPERIMENT FOR {args.subject} (OLD LOADER) ===")
    val_old, te_old = run_experiment(args.subject, use_patched_loader=False, epochs=args.epochs)
    
    print(f"\n=== RUNNING EXPERIMENT FOR {args.subject} (NEW LOADER) ===")
    val_new, te_new = run_experiment(args.subject, use_patched_loader=True, epochs=args.epochs)
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(val_old, label="Old (Buggy)", color='red')
    plt.plot(val_new, label="New (Patched)", color='blue')
    plt.title(f"{args.subject} - Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(te_old, label="Old (Buggy)", color='red')
    plt.plot(te_new, label="New (Patched)", color='blue')
    plt.title(f"{args.subject} - Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    
    out_dir = Path("results/verification")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"label_bug_verification_{args.subject}.png"
    plt.savefig(out_file)
    print(f"\nSaved comparison plot to {out_file}")
    
    diff = max(te_new) - max(te_old)
    print(f"Max Test Acc Change: {diff:.3f} (Old: {max(te_old):.3f}, New: {max(te_new):.3f})")
