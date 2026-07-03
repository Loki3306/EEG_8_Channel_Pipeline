import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import scipy.io
import scipy.signal
import glob
import math

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Models
from models.aad_conformer import AADConformer
from models.eegnet import EEGNet
from models.eegnet_tcn import EEGNetTCN
from models.atcnet import ATCNet
from models.eeg_inception import EEGInception

from training.phase29_cross_subject_train import WindowedDataset, load_aasd_subject
from training.phase30_within_subject_train import NegativePearsonLoss
from training.train_conformer_loso import safe_corr_np

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    
    # 1. Dataset Loading (Identical for all models)
    print("\n--- 1. Loading Common Dataset ---")
    mat_files = glob.glob('/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG/*/*.mat')
    if not mat_files:
        print("ERROR: No .mat files found. Please run on Kaggle.")
        return

    subject = "S18"
    sub_str = f"{subject}.mat"
    sub_path = next((p for p in mat_files if sub_str in p), None)
    if not sub_path:
        print("Subject not found")
        return

    b, a = scipy.signal.butter(4, [1.0/64.0, 8.0/64.0], btype='band')
    sel_idx = [23, 28, 22, 41, 36, 0, 40, 25] # fallback map
    audio_dir = '/kaggle/input/datasets/lokeshgile/aasd-audio-gammatones'

    trials = load_aasd_subject(sub_path, b, a, sel_idx, audio_dir)
    print(f"Loaded {len(trials)} trials for {subject}")

    split_idx = int(0.8 * len(trials))
    train_trials = trials[:split_idx]
    test_trials = trials[split_idx:]

    train_ds = WindowedDataset(train_trials, window_len=128, hop_len=64, censor_margin=256)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    
    # 2. Model Registry
    models = {
        "EEGNet": EEGNet(in_channels=8),
        "EEGNet+TCN": EEGNetTCN(in_channels=8),
        "EEGTCNet(ATC)": ATCNet(in_channels=8),
        "EEG-Inception": EEGInception(in_channels=8),
        "AADConformer (Baseline)": AADConformer(in_channels=8)
    }
    
    results = []
    
    print("\n--- 2. Commencing Benchmark ---")
    epochs = 50
    
    for name, model in models.items():
        print(f"\n==================================================")
        print(f"Evaluating: {name}")
        print(f"==================================================")
        
        model = model.to(device)
        params = count_parameters(model)
        print(f"Parameters: {params:,}")
        
        # Identical Optimizer & Loss
        lr = 1e-3
        weight_decay = 1e-4
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = NegativePearsonLoss().to(device)
        
        # Training Phase
        start_time = time.time()
        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item() * batch_x.size(0)
            
            if epoch % 5 == 0 or epoch == epochs:
                print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss/len(train_ds):.4f}")
        train_time = time.time() - start_time
        
        # Evaluation Phase
        model.eval()
        
        # Train Eval
        train_att, train_unatt = [], []
        with torch.no_grad():
            for t in train_trials:
                eeg = t["eeg"].unsqueeze(0).to(device)
                pred = model(eeg).squeeze(0).cpu().numpy()
                train_att.append(safe_corr_np(pred, t["att"].squeeze(0).numpy()))
                train_unatt.append(safe_corr_np(pred, t["unatt"].squeeze(0).numpy()))
        
        train_pearson = np.mean(train_att)
        train_acc = np.mean(np.array(train_att) > np.array(train_unatt))
        
        # Test Eval
        test_att, test_unatt = [], []
        inf_start = time.time()
        with torch.no_grad():
            for t in test_trials:
                eeg = t["eeg"].unsqueeze(0).to(device)
                pred = model(eeg).squeeze(0).cpu().numpy()
                test_att.append(safe_corr_np(pred, t["att"].squeeze(0).numpy()))
                test_unatt.append(safe_corr_np(pred, t["unatt"].squeeze(0).numpy()))
        inf_time = time.time() - inf_start
        
        test_pearson = np.mean(test_att)
        test_acc = np.mean(np.array(test_att) > np.array(test_unatt))
        
        print(f"\n--- Results for {name} ---")
        print(f"Train Pearson: {train_pearson:.4f} | Train Acc: {train_acc*100:.1f}%")
        print(f"Test Pearson : {test_pearson:.4f} | Test Acc : {test_acc*100:.1f}%")
        print(f"Train Time   : {train_time:.1f}s | Inf Time: {inf_time:.3f}s")
        
        results.append({
            "Model": name,
            "Params": params,
            "Train P": train_pearson,
            "Train Acc": train_acc,
            "Test P": test_pearson,
            "Test Acc": test_acc,
            "Time(s)": train_time
        })
        
    print("\n\n" + "="*80)
    print("PHASE 31: AASD ARCHITECTURE BENCHMARK SUMMARY")
    print("="*80)
    print(f"{'Model':<25} | {'Params':<10} | {'Trn P':<8} | {'Trn Acc':<8} | {'Test P':<8} | {'Test Acc':<8} | {'Time(s)':<8}")
    print("-" * 80)
    
    # Sort by Test Acc (desc), then Test P (desc)
    results.sort(key=lambda x: (x["Test Acc"], x["Test P"]), reverse=True)
    
    for r in results:
        print(f"{r['Model']:<25} | {r['Params']:<10,} | {r['Train P']:<8.4f} | {r['Train Acc']*100:>5.1f}% | {r['Test P']:<8.4f} | {r['Test Acc']*100:>5.1f}% | {r['Time(s)']:<8.1f}")
    
if __name__ == "__main__":
    run_benchmark()
