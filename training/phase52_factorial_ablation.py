import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from scipy.signal import butter, filtfilt
import numpy as np
from pathlib import Path
import sys
import os

# Add parent to path for relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.eegnet_classifier import EEGNetClassifier

# -------------------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------------------
SR = 128
WIN_SEC = 2.0
WIN_SAMPLES = int(WIN_SEC * SR)
COGNITIVE_GAP_SEC = 1.5
GAP_SAMPLES = int(COGNITIVE_GAP_SEC * SR)
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3

BANDS = {
    "Broadband": (1.0, 64.0),
    "Delta": (1.0, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 14.0),
    "Beta": (15.0, 30.0)
}

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    # Scipy expects highcut < nyq. If highcut == nyq, butter throws an error.
    high = min(highcut, nyq - 0.1) / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Applies a zero-phase Butterworth bandpass filter.
    data shape: (channels, time)
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, axis=-1)
    return y

def get_attended_speaker_at_time(start_idx, switch_points):
    current_spk = 'L'
    for spk, idx in switch_points:
        if idx <= start_idx:
            current_spk = spk
        else:
            break
    return current_spk

def extract_windows_for_trial(trial, lowcut, highcut):
    """
    Applies the specified bandpass filter and slices into 2.0s windows.
    Keeps all 60 channels.
    """
    eeg = trial['eeg'].numpy() # (60, Time)
    switch_points = trial['meta']['switch_points']
    
    # 1. APPLY BANDPASS FILTER FIRST (Avoids edge artifacts)
    eeg = apply_bandpass_filter(eeg, lowcut, highcut, SR, order=4)
    
    # 2. Z-Score the filtered EEG per trial
    eeg = (eeg - np.mean(eeg, axis=1, keepdims=True)) / (np.std(eeg, axis=1, keepdims=True) + 1e-8)
    
    T = eeg.shape[1]
    
    X_list = []
    y_list = []
    
    for start in range(0, T - WIN_SAMPLES + 1, WIN_SAMPLES):
        end = start + WIN_SAMPLES
        
        # Check overlap with ANY cognitive gap
        overlap = False
        for spk, sw_idx in switch_points:
            gap_start = sw_idx
            gap_end = sw_idx + GAP_SAMPLES
            if max(start, gap_start) < min(end, gap_end):
                overlap = True
                break
                
        if overlap:
            continue
            
        att_spk = get_attended_speaker_at_time(start, switch_points)
        label = 1.0 if att_spk == 'L' else 0.0
        
        X_list.append(eeg[:, start:end])
        y_list.append(label)
        
    if len(X_list) == 0:
        return None, None
        
    return np.stack(X_list), np.array(y_list)

def main():
    cache_path = Path('/kaggle/working/eeg_cache/S1_processed.pt')
    if not cache_path.exists():
        print("Cache not found. Please run generate_aasd_cache.py first.")
        return
        
    print("\n=======================================================")
    print(" PHASE 52: FACTORIAL FREQUENCY ABLATION (60 CHANNELS)")
    print("=======================================================\n")
    
    print("Loading S1 Cache...")
    cached = torch.load(cache_path, map_location='cpu', weights_only=False)
    trials = cached['raw']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on Device: {device}\n")
    
    results = {}
    
    for band_name, (lowcut, highcut) in BANDS.items():
        print(f"\n=======================================")
        print(f" EVALUATING BAND: {band_name} ({lowcut}-{highcut} Hz)")
        print(f"=======================================")
        
        trial_Xs = []
        trial_ys = []
        
        for tr in trials:
            X, y = extract_windows_for_trial(tr, lowcut, highcut)
            trial_Xs.append(X)
            trial_ys.append(y)
            
        kf = KFold(n_splits=5, shuffle=False)
        fold_aurocs = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(trials)):
            X_train = np.concatenate([trial_Xs[i] for i in train_idx if trial_Xs[i] is not None], axis=0)
            y_train = np.concatenate([trial_ys[i] for i in train_idx if trial_ys[i] is not None], axis=0)
            
            X_test = np.concatenate([trial_Xs[i] for i in test_idx if trial_Xs[i] is not None], axis=0)
            y_test = np.concatenate([trial_ys[i] for i in test_idx if trial_ys[i] is not None], axis=0)
            
            train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            
            test_ds = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
            
            # Using all 60 channels to isolate frequency impact perfectly
            model = EEGNetClassifier(in_channels=60, samples=WIN_SAMPLES).to(device)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
            
            best_val_auc = 0.50
            
            for epoch in range(EPOCHS):
                model.train()
                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                model.eval()
                all_preds = []
                all_labels = []
                with torch.no_grad():
                    for batch_x, batch_y in test_loader:
                        batch_x = batch_x.to(device)
                        logits = model(batch_x)
                        probs = torch.sigmoid(logits).cpu().numpy()
                        all_preds.extend(probs)
                        all_labels.extend(batch_y.numpy())
                        
                try:
                    val_auc = roc_auc_score(all_labels, all_preds)
                except ValueError:
                    val_auc = 0.50
                    
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    
            print(f"  [{band_name}] Fold {fold+1} Best AUROC: {best_val_auc:.4f}")
            fold_aurocs.append(best_val_auc)
            
        avg_auc = np.mean(fold_aurocs)
        print(f"  -> {band_name} Average AUROC: {avg_auc:.4f}")
        results[band_name] = avg_auc

    print("\n=======================================================")
    print(" FACTORIAL ABLATION RESULTS (60 CHANNELS)")
    print("=======================================================")
    for band_name, auc in results.items():
        print(f"  {band_name:<12}: {auc:.4f} AUROC")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
