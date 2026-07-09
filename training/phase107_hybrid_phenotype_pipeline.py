import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np
from scipy import signal
from pathlib import Path
import sys
import os
import random
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.aad_tcn import DeepMatchMismatchTCN

# -------------------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------------------
SR = 128
WIN_SEC = 2.0
HOP_SEC = 0.5
EXCLUSION_SEC = 1.5   
SEQ_SEC = 3.5         

WIN_SAMPLES = int(WIN_SEC * SR)
HOP_SAMPLES = int(HOP_SEC * SR)
EXCLUSION_SAMPLES = int(EXCLUSION_SEC * SR)
SEQ_SAMPLES = int(SEQ_SEC * SR)

EAR_CHANNEL_INDICES = [23, 31, 32, 40, 14, 22, 41, 49]
BATCH_SIZE = 32

# Clinical Calibration Constants
CALIB_EPOCHS = 2       # Extremely fast 2-epoch hyperparameter search
TRAIN_EPOCHS = 12      # Full deployment model training

# Test on 6 biologically diverse subjects to validate the automated routing
TARGET_SUBJECTS = ['S05', 'S08', 'S10', 'S11', 'S13', 'S16']

# The biological phenotypes to evaluate during clinical calibration
CANDIDATE_BANDS = [
    ("Syllabic (<4Hz)", None, 4.0),
    ("Phonemic (4-8Hz)", 4.0, 8.0),
    ("Cortical (8-16Hz)", 8.0, 16.0),
    ("Transients (>16Hz)", 16.0, None),
    ("Broadband (0.5-32Hz)", 0.5, 32.0)
]

def apply_modulation_filter(env, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    if lowcut is None and highcut is not None:
        b, a = signal.butter(order, highcut / nyq, btype='low')
    elif highcut is None and lowcut is not None:
        b, a = signal.butter(order, lowcut / nyq, btype='high')
    else:
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        
    filtered = signal.filtfilt(b, a, env, axis=1)
    return filtered

class MatchMismatchDataset(Dataset):
    def __init__(self, sequences):
        self.seqs = sequences
    def __len__(self): return len(self.seqs)
    def __getitem__(self, idx):
        e, a, y = self.seqs[idx]
        return e, a, y[-1]

def extract_match_mismatch_sequences(trials):
    sequences = []
    seq_hop = int(0.5 * SR) 
    
    for tr in trials:
        eeg = tr['eeg']
        env_l = tr['env_l'] 
        env_r = tr['env_r']
        sp = tr['meta']['switch_points']
        T = eeg.shape[1]
        
        boundaries = [0]
        boundaries.extend([idx for spk, idx in sp])
        boundaries = sorted(set(boundaries))
        if boundaries[-1] != T: boundaries.append(T)
            
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i+1]
            current_spk = 'L'
            for spk, idx in sp:
                if idx <= start_idx: current_spk = spk
                else: break
                
            safe_start = start_idx + EXCLUSION_SAMPLES
            safe_end = end_idx
            
            if safe_end - safe_start >= SEQ_SAMPLES:
                for seq_start in range(safe_start, safe_end - SEQ_SAMPLES + 1, seq_hop):
                    e_seq = eeg[:, seq_start:seq_start + SEQ_SAMPLES]
                    al_seq = env_l[:, seq_start:seq_start + SEQ_SAMPLES]
                    ar_seq = env_r[:, seq_start:seq_start + SEQ_SAMPLES]
                    
                    e = torch.from_numpy(e_seq.copy()).unfold(-1, WIN_SAMPLES, HOP_SAMPLES).permute(1, 0, 2)
                    al = torch.from_numpy(al_seq.copy()).unfold(-1, WIN_SAMPLES, HOP_SAMPLES).permute(1, 0, 2)
                    ar = torch.from_numpy(ar_seq.copy()).unfold(-1, WIN_SAMPLES, HOP_SAMPLES).permute(1, 0, 2)
                    
                    num_windows = e.shape[0]
                    
                    if current_spk == 'L':
                        match_aud = al
                        mismatch_aud = ar
                    else:
                        match_aud = ar
                        mismatch_aud = al
                        
                    y_match = torch.full((num_windows,), 1.0, dtype=torch.float32)
                    sequences.append((e, match_aud, y_match))
                    
                    y_mismatch = torch.full((num_windows,), 0.0, dtype=torch.float32)
                    sequences.append((e, mismatch_aud, y_mismatch))
                    
    return sequences

def get_trial_dominant_speaker(tr):
    sp = tr['meta']['switch_points']
    T = tr['eeg'].shape[1]
    
    boundaries = [0]
    boundaries.extend([idx for spk, idx in sp])
    boundaries = sorted(set(boundaries))
    if boundaries[-1] != T: boundaries.append(T)
        
    l_duration = 0
    r_duration = 0
    
    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i+1]
        current_spk = 'L'
        for spk, idx in sp:
            if idx <= start_idx: current_spk = spk
            else: break
            
        if current_spk == 'L': l_duration += (end_idx - start_idx)
        else: r_duration += (end_idx - start_idx)
        
    return 'L' if l_duration >= r_duration else 'R'

def stratified_trial_split(trials, train_ratio=0.8):
    l_trials = []
    r_trials = []
    
    for i, tr in enumerate(trials):
        if get_trial_dominant_speaker(tr) == 'L':
            l_trials.append(i)
        else:
            r_trials.append(i)
            
    random.seed(42)
    random.shuffle(l_trials)
    random.shuffle(r_trials)
    
    l_split = int(len(l_trials) * train_ratio)
    r_split = int(len(r_trials) * train_ratio)
    
    train_indices = l_trials[:l_split] + r_trials[:r_split]
    eval_indices = l_trials[l_split:] + r_trials[r_split:]
    
    random.shuffle(train_indices)
    random.shuffle(eval_indices)
    
    return train_indices, eval_indices

def fast_clinical_calibration(raw_train_trials, device):
    """
    Simulates a 60-second clinical hearing-aid fitting session.
    Automatically evaluates the candidate modulation bands on the train set
    to identify the subject's mathematically optimal biological phenotype.
    """
    print(f"\n  [Calibration] Initiating fast phenotype sweep...", flush=True)
    start_time = time.time()
    
    # Internal split for calibration validation using Strict Stratification!
    calib_train_idx, calib_val_idx = stratified_trial_split(raw_train_trials, train_ratio=0.8)
    
    best_band = None
    best_val_auc = 0
    band_results = {}
    
    for band_name, lowcut, highcut in CANDIDATE_BANDS:
        # 1. Filter the raw data to the candidate band
        band_train_trials = []
        band_val_trials = []
        
        for i, tr in enumerate(raw_train_trials):
            eeg = tr['eeg']
            env_l = apply_modulation_filter(tr['env_l'], lowcut, highcut, SR)
            env_r = apply_modulation_filter(tr['env_r'], lowcut, highcut, SR)
            
            # Normalize AFTER filtering (to match deployment exactly)
            eeg = (eeg - np.mean(eeg, axis=1, keepdims=True)) / (np.std(eeg, axis=1, keepdims=True) + 1e-8)
            env_l = (env_l - np.mean(env_l, axis=1, keepdims=True)) / (np.std(env_l, axis=1, keepdims=True) + 1e-8)
            env_r = (env_r - np.mean(env_r, axis=1, keepdims=True)) / (np.std(env_r, axis=1, keepdims=True) + 1e-8)
            
            trial = {'eeg': eeg, 'env_l': env_l, 'env_r': env_r, 'meta': tr['meta']}
            if i in calib_train_idx:
                band_train_trials.append(trial)
            else:
                band_val_trials.append(trial)
                
        # 2. Extract sequences
        c_train_seq = extract_match_mismatch_sequences(band_train_trials)
        c_val_seq = extract_match_mismatch_sequences(band_val_trials)
        
        train_loader = DataLoader(MatchMismatchDataset(c_train_seq), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)
        val_loader = DataLoader(MatchMismatchDataset(c_val_seq), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=2)
        
        # 3. Train tiny model for 2 epochs
        model = DeepMatchMismatchTCN(eeg_channels=8, latent_dim=64, tcn_channels=[64, 64, 64], kernel_size=2, dropout=0.2, encoder_type='baseline', audio_channels=16).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        scaler = torch.amp.GradScaler(device.type, enabled=(device.type == 'cuda'))
        
        band_best_auc = 0
        for epoch in range(CALIB_EPOCHS):
            model.train()
            for b_e, b_a, b_y in train_loader:
                b_e, b_a, b_y = b_e.to(device, non_blocking=True).float(), b_a.to(device, non_blocking=True).float(), b_y.to(device, non_blocking=True).float()
                if b_e.size(0) == 1: continue 
                optimizer.zero_grad()
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                    logits, _ = model(b_e, b_a)
                    loss = criterion(logits, b_y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for b_e, b_a, b_y in val_loader:
                    b_e, b_a = b_e.to(device, non_blocking=True).float(), b_a.to(device, non_blocking=True).float()
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                        logits, _ = model(b_e, b_a)
                    all_preds.extend(torch.sigmoid(logits).cpu().numpy().flatten())
                    all_labels.extend(b_y.numpy().flatten())
            
            if len(np.unique(all_labels)) > 1:
                auc = roc_auc_score(all_labels, all_preds)
                is_inverted = (auc < 0.5)
                effective_auc = (1.0 - auc) if is_inverted else auc
                if effective_auc > band_best_auc:
                    band_best_auc = effective_auc
                    band_invert = is_inverted
                
        band_results[band_name] = band_best_auc
        print(f"    - {band_name}: {band_best_auc:.4f} {'(Inverted)' if band_invert else ''}")
        
        if band_best_auc > best_val_auc:
            best_val_auc = band_best_auc
            best_band = (band_name, lowcut, highcut)
            calibration_invert = band_invert
            
    print(f"  [Calibration] Complete in {time.time()-start_time:.1f}s.")
    print(f"  [Calibration] Winner: >> {best_band[0]} << (Invert: {calibration_invert})", flush=True)
    return best_band, calibration_invert

def main():
    possible_paths = [
        Path('/kaggle/input/datasets/lowkieee/multiband-cache/kaggle/working/multiband_cache'),
        Path('/kaggle/input/multiband-cache/kaggle/working/multiband_cache'),
        Path('/kaggle/input/aasd-universal-cache-v1/kaggle/working/multiband_cache'),
        Path('/kaggle/working/multiband_cache')
    ]
    
    cache_dir = Path('/kaggle/working/multiband_cache')
    for p in possible_paths:
        if p.exists() and len(list(p.glob('*_multiband.pt'))) > 0:
            cache_dir = p
            break
            
    print(f"\n=======================================================")
    print(f" PHASE 107: HYBRID PHENOTYPE DEPLOYMENT PIPELINE")
    print(f" 1. Fast Clinical Calibration (Find optimal band)")
    print(f" 2. Target Band Deployment Training")
    print(f"=======================================================\n", flush=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on Device: {device}\n", flush=True)
    
    cache_files = sorted(list(cache_dir.glob('*_multiband.pt')))
    filtered_files = [f for f in cache_files if f.stem.split('_')[0] in TARGET_SUBJECTS]
    
    final_results = {}
    
    for cache_file in filtered_files:
        subj_name = cache_file.stem.split('_')[0]
        print(f"\n=======================================================")
        print(f" SUBJECT {subj_name}")
        print(f"=======================================================", flush=True)
        
        cached = torch.load(cache_file, map_location='cpu', weights_only=False)['raw']
        
        # Load raw pristine data
        raw_trials = []
        for i in range(len(cached)):
            tr = cached[i]
            eeg = tr['eeg'].numpy()[EAR_CHANNEL_INDICES, :] 
            env_l = tr['env_l'].numpy()
            env_r = tr['env_r'].numpy()
            
            # Apply Temporal Modulation Filter FIRST (Phase 107 GPT Fix)
            # We don't filter raw_trials yet because candidate bands will dynamically filter them.
            # So just store raw pristine arrays, but we don't normalize here anymore!
            
            min_len = min(eeg.shape[1], env_l.shape[1])
            raw_trials.append({
                'eeg': eeg[:, :min_len], 
                'env_l': env_l[:, :min_len], 
                'env_r': env_r[:, :min_len], 
                'meta': tr['meta']
            })
            
        # STRICT STRATIFIED TRIAL-LEVEL SPLIT
        # This absolutely guarantees equal L/R representation in Train and Eval
        train_indices, eval_indices = stratified_trial_split(raw_trials, train_ratio=0.8)
        
        raw_train_trials = [raw_trials[i] for i in train_indices]
        raw_eval_trials = [raw_trials[i] for i in eval_indices]
        
        # 1. FAST CLINICAL CALIBRATION
        best_band_info, calibration_invert = fast_clinical_calibration(raw_train_trials, device)
        best_band_name, lowcut, highcut = best_band_info
        
        # 2. APPLY WINNING PHENOTYPE
        print(f"\n  [Deployment] Extracting {best_band_name} sequences for deployment...", flush=True)
        final_train_trials = []
        final_eval_trials = []
        
        for tr in raw_train_trials:
            eeg = tr['eeg']
            env_l = apply_modulation_filter(tr['env_l'], lowcut, highcut, SR)
            env_r = apply_modulation_filter(tr['env_r'], lowcut, highcut, SR)
            
            # Normalize AFTER filtering (GPT Methodological Fix)
            eeg = (eeg - np.mean(eeg, axis=1, keepdims=True)) / (np.std(eeg, axis=1, keepdims=True) + 1e-8)
            env_l = (env_l - np.mean(env_l, axis=1, keepdims=True)) / (np.std(env_l, axis=1, keepdims=True) + 1e-8)
            env_r = (env_r - np.mean(env_r, axis=1, keepdims=True)) / (np.std(env_r, axis=1, keepdims=True) + 1e-8)
            
            final_train_trials.append({'eeg': eeg, 'env_l': env_l, 'env_r': env_r, 'meta': tr['meta']})
            
        for tr in raw_eval_trials:
            eeg = tr['eeg']
            env_l = apply_modulation_filter(tr['env_l'], lowcut, highcut, SR)
            env_r = apply_modulation_filter(tr['env_r'], lowcut, highcut, SR)
            
            eeg = (eeg - np.mean(eeg, axis=1, keepdims=True)) / (np.std(eeg, axis=1, keepdims=True) + 1e-8)
            env_l = (env_l - np.mean(env_l, axis=1, keepdims=True)) / (np.std(env_l, axis=1, keepdims=True) + 1e-8)
            env_r = (env_r - np.mean(env_r, axis=1, keepdims=True)) / (np.std(env_r, axis=1, keepdims=True) + 1e-8)
            
            final_eval_trials.append({'eeg': eeg, 'env_l': env_l, 'env_r': env_r, 'meta': tr['meta']})
            
        train_seqs = extract_match_mismatch_sequences(final_train_trials)
        eval_seqs = extract_match_mismatch_sequences(final_eval_trials)
        
        if len(train_seqs) == 0 or len(eval_seqs) == 0:
            print("Not enough sequences extracted.")
            continue
            
        train_loader = DataLoader(MatchMismatchDataset(train_seqs), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)
        eval_loader = DataLoader(MatchMismatchDataset(eval_seqs), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=2)
        
        # 3. DEPLOYMENT TRAINING
        print(f"  [Deployment] Training Full TCN on chosen phenotype...", flush=True)
        model = DeepMatchMismatchTCN(eeg_channels=8, latent_dim=64, tcn_channels=[64, 64, 64], kernel_size=2, dropout=0.2, encoder_type='baseline', audio_channels=16).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        scaler = torch.amp.GradScaler(device.type, enabled=(device.type == 'cuda'))
        
        deployment_best_auc = 0
        inverted_diagnostic = False
        
        for epoch in range(TRAIN_EPOCHS):
            model.train()
            for b_e, b_a, b_y in train_loader:
                b_e, b_a, b_y = b_e.to(device, non_blocking=True).float(), b_a.to(device, non_blocking=True).float(), b_y.to(device, non_blocking=True).float()
                if b_e.size(0) == 1: continue 
                optimizer.zero_grad()
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                    logits, _ = model(b_e, b_a)
                    loss = criterion(logits, b_y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            # We only evaluate on the eval set AFTER all epochs finish to prevent optimistic bias!
            # The final epoch is the true deployment model checkpoint.
            
        # 4. FINAL DEPLOYMENT EVALUATION (Strictly 1 look)
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for b_e, b_a, b_y in eval_loader:
                b_e, b_a = b_e.to(device, non_blocking=True).float(), b_a.to(device, non_blocking=True).float()
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                    logits, _ = model(b_e, b_a)
                all_preds.extend(torch.sigmoid(logits).cpu().numpy().flatten())
                all_labels.extend(b_y.numpy().flatten())
                
        deployment_best_auc = 0
        if len(np.unique(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_preds)
            # Apply strictly the inversion policy learned during calibration!
            deployment_best_auc = (1.0 - auc) if calibration_invert else auc
            
        print(f"  [Deployment] Final Deployment AUROC: {deployment_best_auc:.4f} {'(Inverted via Calibration)' if calibration_invert else ''}")
        final_results[subj_name] = {'Band': best_band_name, 'AUROC': deployment_best_auc}

    print("\n\n=======================================================")
    print(" PHASE 107 HYBRID PIPELINE RESULTS")
    print("=======================================================")
    print(f"{'Subject':<10} {'Automatically Routed Band':<25} {'Deployment AUROC':<15}")
    for subj in TARGET_SUBJECTS:
        if subj in final_results:
            print(f"{subj:<10} {final_results[subj]['Band']:<25} {final_results[subj]['AUROC']:.4f}")

if __name__ == "__main__":
    main()
