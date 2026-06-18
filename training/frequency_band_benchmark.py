"""
frequency_band_benchmark.py
Efficient Mini-LOSO frequency band benchmark.
Tests EEG frequency variants with zero disk I/O during training.
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from baselines.ridge_aad import subject_files, load_subject_examples
from models.matchnet import ContrastiveMatchNet

FS = 64.0
MINI_LOSO_SUBJECTS = ['S1', 'S4', 'S6', 'S8', 'S11', 'S14']
CHANNELS = [13, 46, 43, 23, 50, 0, 52, 14]
NUM_BANDS = 28 # Gammatone

VARIANTS = {
    'A': {'low': 0.1, 'high': None, 'name': '0.1 Hz HP (Baseline)'},
    'B': {'low': 1.0, 'high': 8.0, 'name': '1-8 Hz (Current)'},
    'C': {'low': 1.0, 'high': 12.0, 'name': '1-12 Hz'},
    'D': {'low': 4.0, 'high': 8.0, 'name': '4-8 Hz (Theta)'},
    'E': {'low': 8.0, 'high': 12.0, 'name': '8-12 Hz (Alpha)'},
    'F': {'low': 12.0, 'high': 30.0, 'name': '12-30 Hz (Beta)'},
    'G': {'low': 1.0, 'high': 30.0, 'name': '1-30 Hz'},
    'H': {'low': 4.0, 'high': 30.0, 'name': '4-30 Hz'}
}

def print_and_write(f_out, text):
    print(text)
    f_out.write(text + "\n")

def filter_eeg(data, lowcut, highcut, fs=64.0):
    nyq = 0.5 * fs
    low = lowcut / nyq
    if highcut is None:
        b, a = butter(4, low, btype='high')
    else:
        high = min(highcut, 31.9) / nyq
        b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, data, axis=1)

def normalize_array(arr):
    arr = arr - arr.mean(axis=0, keepdims=True)
    scale = arr.std(axis=0, keepdims=True) + 1e-12
    return arr / scale

def get_mapping_data():
    map_file = REPO_ROOT / "data" / "audio_mapping.json"
    env_file = Path("/kaggle/input/datasets/lokeshgile/gammatone-envelope/gammatone_envelopes.pkl")
    if not env_file.exists():
        env_file = REPO_ROOT / "data" / "gammatone_envelopes.pkl"
    with open(map_file, 'r') as f:
        mapping = json.load(f)
    with open(env_file, 'rb') as f:
        envelopes = pickle.load(f)
    return mapping, envelopes

def compute_psd_stats(eeg_data):
    # eeg_data: list of arrays (channels, time)
    all_psds = []
    f_psd = None
    for eeg in eeg_data:
        f, pxx = welch(eeg, fs=FS, nperseg=int(FS*2), axis=1)
        if f_psd is None: f_psd = f
        all_psds.append(np.mean(pxx, axis=0)) # mean over channels
    
    avg_psd = np.mean(all_psds, axis=0)
    
    # Calculate power %
    total_power = np.trapz(avg_psd[f_psd <= 30], f_psd[f_psd <= 30])
    bands = {'Delta': (1,4), 'Theta': (4,8), 'Alpha': (8,12), 'Beta': (12,30)}
    stats = {}
    for name, (low, high) in bands.items():
        idx = np.logical_and(f_psd >= low, f_psd <= high)
        power = np.trapz(avg_psd[idx], f_psd[idx])
        stats[name] = (power / total_power) * 100 if total_power > 0 else 0
    return stats

def chunk_trial(x, ya, yb, label, window_sec, hop_sec):
    win_samples = int(window_sec * FS)
    hop_samples = int(hop_sec * FS)
    n_chunks = (x.shape[1] - win_samples) // hop_samples + 1
    
    X_chunks, YA_chunks, YB_chunks, L_chunks = [], [], [], []
    for i in range(n_chunks):
        start = i * hop_samples
        end = start + win_samples
        X_chunks.append(x[:, start:end])
        YA_chunks.append(ya[:, start:end])
        YB_chunks.append(yb[:, start:end])
        L_chunks.append(label)
    return X_chunks, YA_chunks, YB_chunks, L_chunks

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, ya, yb, labels in dataloader:
            x = x.to(device)
            ya = ya.to(device)
            yb = yb.to(device)
            out_a, out_b = model(x, ya, yb)
            # MatchNet uses distance
            d_a = torch.norm(out_a, dim=1)
            d_b = torch.norm(out_b, dim=1)
            preds = (d_a > d_b).int().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return accuracy_score(all_labels, all_preds)

def build_benchmark(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "frequency_band_benchmark_report.md"
    csv_path = out_dir / "frequency_band_results.csv"
    
    mapping, envelopes = get_mapping_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Preload and Cache
    print("Preloading and caching datasets in memory...")
    all_paths = subject_files()
    subject_paths = [p for p in all_paths if p.stem.split('_')[0] in MINI_LOSO_SUBJECTS]
    
    if not subject_paths:
        print("ERROR: No subjects found.")
        return
        
    cached_variants = {v: {} for v in VARIANTS}
    psd_stats = []
    
    # For chunking
    WINS = [2, 5]
    HOP = 0.5
    
    # Cache structure: cached_variants[variant][subject][window] = (X, YA, YB, L)
    for p in subject_paths:
        sub_key = p.stem.split("_")[0]
        print(f"  Loading {sub_key}...")
        examples = load_subject_examples(p)
        
        # Compute PSD on unfiltered EEG
        eegs = [ex.eeg[:, CHANNELS] for ex in examples]
        stats = compute_psd_stats(eegs)
        stats['Subject'] = sub_key
        psd_stats.append(stats)
        
        for v_key, v_params in VARIANTS.items():
            cached_variants[v_key][sub_key] = {w: ({'X':[],'YA':[],'YB':[],'L':[]}) for w in WINS}
            
            for i, ex in enumerate(examples):
                eeg = ex.eeg[:, CHANNELS].T # (channels, time)
                eeg_filt = filter_eeg(eeg, v_params['low'], v_params['high'])
                x_norm = normalize_array(eeg_filt.T).T
                
                trial_key = f"trial_{i}"
                if sub_key in mapping and trial_key in mapping[sub_key]:
                    fname_a = mapping[sub_key][trial_key]["wavA"]["filename"]
                    fname_b = mapping[sub_key][trial_key]["wavB"]["filename"]
                    env_a_full = envelopes[fname_a] 
                    env_b_full = envelopes[fname_b] 
                else:
                    continue
                    
                min_len = min(x_norm.shape[1], env_a_full.shape[1])
                x_norm = x_norm[:, :min_len]
                env_a = normalize_array(env_a_full[:, :min_len].T).T
                env_b = normalize_array(env_b_full[:, :min_len].T).T
                
                label = ex.label
                
                for w in WINS:
                    xc, yac, ybc, lc = chunk_trial(x_norm, env_a, env_b, label, window_sec=w, hop_sec=HOP)
                    cached_variants[v_key][sub_key][w]['X'].extend(xc)
                    cached_variants[v_key][sub_key][w]['YA'].extend(yac)
                    cached_variants[v_key][sub_key][w]['YB'].extend(ybc)
                    cached_variants[v_key][sub_key][w]['L'].extend(lc)
                    
            # Convert to tensors
            for w in WINS:
                cached_variants[v_key][sub_key][w]['X'] = torch.tensor(np.array(cached_variants[v_key][sub_key][w]['X']), dtype=torch.float32)
                cached_variants[v_key][sub_key][w]['YA'] = torch.tensor(np.array(cached_variants[v_key][sub_key][w]['YA']), dtype=torch.float32)
                cached_variants[v_key][sub_key][w]['YB'] = torch.tensor(np.array(cached_variants[v_key][sub_key][w]['YB']), dtype=torch.float32)
                cached_variants[v_key][sub_key][w]['L'] = torch.tensor(np.array(cached_variants[v_key][sub_key][w]['L']), dtype=torch.long)
                
    # 2. Train and Evaluate
    print("\nStarting Mini-LOSO Training...")
    results = []
    
    for v_key, v_params in VARIANTS.items():
        v_name = v_params['name']
        print(f"\nEvaluating Variant {v_key}: {v_name}")
        
        variant_accs_2s = []
        variant_accs_5s = []
        
        for test_sub in MINI_LOSO_SUBJECTS:
            print(f"  Testing on {test_sub}")
            
            # Build training datasets
            train_X, train_YA, train_YB, train_L = [], [], [], []
            for sub in MINI_LOSO_SUBJECTS:
                if sub != test_sub:
                    train_X.append(cached_variants[v_key][sub][2]['X'])
                    train_YA.append(cached_variants[v_key][sub][2]['YA'])
                    train_YB.append(cached_variants[v_key][sub][2]['YB'])
                    train_L.append(cached_variants[v_key][sub][2]['L'])
                    
            train_X = torch.cat(train_X)
            train_YA = torch.cat(train_YA)
            train_YB = torch.cat(train_YB)
            train_L = torch.cat(train_L)
            
            # Use 2s for training
            train_ds = TensorDataset(train_X, train_YA, train_YB, train_L)
            train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
            
            model = ContrastiveMatchNet(eeg_model_type='vlaai_lite', eeg_channels=len(CHANNELS), audio_channels=NUM_BANDS, latent_dim=64, audio_model_type='standard').to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            
            # Train for 15 epochs (fixed, no validation stopping needed for this benchmark)
            model.train()
            for epoch in range(15):
                for x, ya, yb, labels in train_loader:
                    x, ya, yb, labels = x.to(device), ya.to(device), yb.to(device), labels.to(device)
                    optimizer.zero_grad()
                    out_a, out_b = model(x, ya, yb)
                    # Contrastive loss
                    y_float = labels.float()
                    d_a = torch.norm(out_a, dim=1)
                    d_b = torch.norm(out_b, dim=1)
                    loss = torch.mean(y_float * torch.relu(d_a - d_b + 0.1) + (1 - y_float) * torch.relu(d_b - d_a + 0.1))
                    loss.backward()
                    optimizer.step()
                    
            # Evaluate 2s
            test_2_ds = TensorDataset(cached_variants[v_key][test_sub][2]['X'], cached_variants[v_key][test_sub][2]['YA'], cached_variants[v_key][test_sub][2]['YB'], cached_variants[v_key][test_sub][2]['L'])
            test_2_loader = DataLoader(test_2_ds, batch_size=256, shuffle=False)
            acc_2s = evaluate_model(model, test_2_loader, device)
            
            # Evaluate 5s
            test_5_ds = TensorDataset(cached_variants[v_key][test_sub][5]['X'], cached_variants[v_key][test_sub][5]['YA'], cached_variants[v_key][test_sub][5]['YB'], cached_variants[v_key][test_sub][5]['L'])
            test_5_loader = DataLoader(test_5_ds, batch_size=256, shuffle=False)
            acc_5s = evaluate_model(model, test_5_loader, device)
            
            variant_accs_2s.append(acc_2s)
            variant_accs_5s.append(acc_5s)
            
            results.append({
                'Variant': v_name,
                'Test_Subject': test_sub,
                'Acc_2s': acc_2s * 100,
                'Acc_5s': acc_5s * 100
            })
            
    # Compile Results
    df_res = pd.DataFrame(results)
    
    # Calculate means
    mean_res = df_res.groupby('Variant').mean(numeric_only=True).reset_index()
    
    # Baseline for Delta calculation
    baseline_2s = mean_res[mean_res['Variant'] == '1-8 Hz (Current)']['Acc_2s'].values[0]
    baseline_5s = mean_res[mean_res['Variant'] == '1-8 Hz (Current)']['Acc_5s'].values[0]
    baseline_mean = (baseline_2s + baseline_5s) / 2.0
    
    mean_res['Mean_Acc'] = (mean_res['Acc_2s'] + mean_res['Acc_5s']) / 2.0
    mean_res['Delta'] = mean_res['Mean_Acc'] - baseline_mean
    
    # Output to CSV
    mean_res.to_csv(csv_path, index=False)
    
    # Generate Report
    with open(report_path, "w") as f_out:
        print_and_write(f_out, "# Frequency-Band Benchmark Report\n")
        
        print_and_write(f_out, "## 1. PSD Power Distribution")
        df_psd = pd.DataFrame(psd_stats)
        print_and_write(f_out, df_psd.to_markdown(index=False))
        print_and_write(f_out, "\n")
        
        print_and_write(f_out, "## 2. Benchmark Results")
        print_and_write(f_out, mean_res[['Variant', 'Acc_2s', 'Acc_5s', 'Mean_Acc', 'Delta']].to_markdown(index=False))
        print_and_write(f_out, "\n")
        
        # Success Criteria
        best_delta = mean_res['Delta'].max()
        best_variant = mean_res.loc[mean_res['Delta'].idxmax(), 'Variant']
        
        print_and_write(f_out, "## 3. Recommendation")
        if best_delta >= 2.0:
            print_and_write(f_out, f"✅ **SUCCESS**: Variant '{best_variant}' improved performance by +{best_delta:.2f}%. Recommend promoting to full LOSO evaluation.")
        else:
            print_and_write(f_out, f"❌ **FAILURE**: No frequency variant met the +2.0% threshold (Best: {best_variant} at {best_delta:+.2f}%). Recommend stopping further frequency-band exploration.")
            
    print(f"\nBenchmark complete. Report saved to {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="reports")
    args = parser.parse_args()
    build_benchmark(args.out_dir)
