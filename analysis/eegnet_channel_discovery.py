"""
EEGNet-Specific Channel Discovery.

Strategy:
1. Train EEGNet with ALL 64 channels on the 4-subject screening set (LOSO).
2. Extract the spatial depthwise convolution weights from block1.
3. Compute channel importance = mean absolute weight across all spatial filters and folds.
4. Rank all 64 channels by importance.
5. Output Top 4, Top 8, Top 12, Top 16 channel sets.
6. Then screen each subset on the same 4 subjects to find the optimal set.
"""
import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from copy import deepcopy
from scipy.signal import butter, filtfilt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.eegnet import EEGNet
from baselines.ridge_aad import load_subject_examples, subject_files, iter_leave_one_subject_out

# DTU channel ordering (66 channels, indices 0-65)
DTU_CHANNEL_NAMES = [
    "Fp1", "Fpz", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "FC5", "FC1", "FC2", "FC6", "M1", "T7", "C3", "Cz",
    "C4", "T8", "CP5", "CP1", "CP2", "CP6", "M2", "P7",
    "P3", "Pz", "P4", "P8", "POz", "O1", "Oz", "O2",
    "AF7", "AF3", "AF4", "AF8", "F5", "F1", "F2", "F6",
    "FT7", "FC3", "FC4", "FT8", "C5", "C1", "C2", "C6",
    "TP7", "CP3", "CPz", "CP4", "TP8", "P5", "P1", "P2",
    "P6", "PO5", "PO3", "PO4", "PO6", "FT9", "FT10", "FCz",
    "PO7", "PO8"
]

SCREENING_SUBJECTS = ["S1_data_preproc", "S7_data_preproc", "S8_data_preproc", "S14_data_preproc"]
CURRENT_BEST_CHANNELS = [13, 46, 43, 23, 50, 0, 52, 14]

FS = 64
DECISION_WINDOW_SEC = 10

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2, axis=0):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data, axis=axis)
    return y

def normalize_array(arr):
    arr = arr - arr.mean(axis=0, keepdims=True)
    scale = arr.std(axis=0, keepdims=True) + 1e-12
    return arr / scale

class PearsonMSELoss(nn.Module):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        pred_mean = pred.mean(dim=2, keepdim=True)
        target_mean = target.mean(dim=2, keepdim=True)
        pred_std = pred.std(dim=2, keepdim=True) + 1e-8
        target_std = target.std(dim=2, keepdim=True) + 1e-8
        cov = ((pred - pred_mean) * (target - target_mean)).mean(dim=2)
        corr = cov / (pred_std.squeeze(2) * target_std.squeeze(2))
        pearson_loss = 1 - corr.mean()
        mse_loss = self.mse(pred, target)
        return pearson_loss + self.alpha * mse_loss

def prepare_dataset(examples, channels, lowcut, highcut):
    """Prepare dataset using specified channel indices."""
    X, Y, Y_A, Y_B = [], [], [], []
    for ex in examples:
        eeg = ex.eeg[:, channels].T
        eeg = butter_bandpass_filter(eeg, lowcut, highcut, FS, axis=1)
        wav_a = butter_bandpass_filter(ex.wav_a.reshape(-1, 1), lowcut, highcut, FS, axis=0).ravel()
        wav_b = butter_bandpass_filter(ex.wav_b.reshape(-1, 1), lowcut, highcut, FS, axis=0).ravel()
        
        x_norm = normalize_array(eeg.T).T
        env_a = normalize_array(wav_a.reshape(-1, 1)).ravel()
        env_b = normalize_array(wav_b.reshape(-1, 1)).ravel()
        target_env = env_a
        
        min_len = min(x_norm.shape[1], len(target_env))
        X.append(x_norm[:, :min_len])
        Y.append(target_env[:min_len])
        Y_A.append(env_a[:min_len])
        Y_B.append(env_b[:min_len])
    return X, Y, Y_A, Y_B

def evaluate_model(model, X, Y_A, Y_B, device):
    model.eval()
    window_samples = DECISION_WINDOW_SEC * FS
    n_correct = 0.0
    n_total = 0
    with torch.no_grad():
        for i in range(len(X)):
            x = torch.FloatTensor(X[i]).unsqueeze(0).to(device)
            pred = model(x).squeeze(0).squeeze(0).cpu().numpy()
            ea, eb = Y_A[i], Y_B[i]
            start = 0
            while start + window_samples <= len(pred):
                end = start + window_samples
                p = pred[start:end]
                std_p = np.std(p)
                if std_p < 1e-12:
                    ca, cb = 0.0, 0.0
                else:
                    ca = np.corrcoef(p, ea[start:end])[0, 1]
                    cb = np.corrcoef(p, eb[start:end])[0, 1]
                if ca > cb:
                    n_correct += 1.0
                elif ca == cb:
                    n_correct += 0.5
                n_total += 1
                start += window_samples
    return n_correct, n_total

def extract_spatial_importance(model):
    """Extract channel importance from the spatial depthwise conv in block1.
    
    The spatial conv is block1[2]: Conv2d(F1, F1*D, (in_channels, 1), groups=F1)
    Weight shape: [F1*D, 1, in_channels, 1]
    
    Channel importance = mean absolute weight across all F1*D filters.
    """
    spatial_conv = model.block1[2]  # The depthwise conv
    weights = spatial_conv.weight.data.cpu().numpy()  # [F1*D, 1, in_channels, 1]
    weights = weights.squeeze()  # [F1*D, in_channels]
    
    # Importance = mean absolute weight per channel across all filters
    importance = np.mean(np.abs(weights), axis=0)  # [in_channels]
    return importance

def train_and_extract(paths, subject_examples, all_channels, lowcut, highcut, device):
    """Train 64-channel EEGNet on screening subjects and extract channel importance."""
    folds = list(iter_leave_one_subject_out(paths))
    num_channels = len(all_channels)
    
    all_importances = []
    
    for fold_idx, (held_out_path, train_paths) in enumerate(folds):
        held_out_key = str(held_out_path)
        print(f"\n  Fold {fold_idx+1}/{len(folds)}: held-out = {held_out_path.stem}")
        
        train_exs = []
        for p in train_paths:
            train_exs.extend(subject_examples[str(p)])
        
        np.random.seed(42)
        np.random.shuffle(train_exs)
        val_split = int(0.1 * len(train_exs))
        val_exs = train_exs[:val_split]
        train_exs = train_exs[val_split:]
        
        X_tr, Y_tr, _, _ = prepare_dataset(train_exs, all_channels, lowcut, highcut)
        X_va, _, YA_va, YB_va = prepare_dataset(val_exs, all_channels, lowcut, highcut)
        
        model = EEGNet(in_channels=num_channels).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = PearsonMSELoss(alpha=0.1)
        
        best_val_acc = 0.0
        best_weights = deepcopy(model.state_dict())
        patience = 15
        epochs_no_improve = 0
        
        for epoch in range(100):
            model.train()
            train_loss = 0.0
            for i in range(len(X_tr)):
                x = torch.FloatTensor(X_tr[i]).unsqueeze(0).to(device)
                y = torch.FloatTensor(Y_tr[i]).unsqueeze(0).unsqueeze(0).to(device)
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            nc_va, nt_va = evaluate_model(model, X_va, YA_va, YB_va, device)
            val_acc = nc_va / max(nt_va, 1)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            if (epoch + 1) % 10 == 0 or epochs_no_improve >= patience:
                print(f"    Epoch {epoch+1:02d} | Loss: {train_loss:.4f} | Val: {val_acc*100:.1f}% | Pat: {epochs_no_improve}/{patience}")
            
            if epochs_no_improve >= patience:
                break
        
        model.load_state_dict(best_weights)
        importance = extract_spatial_importance(model)
        all_importances.append(importance)
        print(f"    Best val acc: {best_val_acc*100:.1f}%")
    
    # Average importance across folds
    mean_importance = np.mean(all_importances, axis=0)
    return mean_importance

def screen_channel_set(name, channels, paths, subject_examples, lowcut, highcut, device):
    """Train and evaluate EEGNet with a specific channel set on screening subjects."""
    folds = list(iter_leave_one_subject_out(paths))
    all_accs = []
    
    for held_out_path, train_paths in folds:
        held_out_key = str(held_out_path)
        
        train_exs = []
        for p in train_paths:
            train_exs.extend(subject_examples[str(p)])
        test_exs = subject_examples[held_out_key]
        
        np.random.seed(42)
        np.random.shuffle(train_exs)
        val_split = int(0.1 * len(train_exs))
        val_exs = train_exs[:val_split]
        train_exs = train_exs[val_split:]
        
        X_tr, Y_tr, _, _ = prepare_dataset(train_exs, channels, lowcut, highcut)
        X_va, _, YA_va, YB_va = prepare_dataset(val_exs, channels, lowcut, highcut)
        X_te, _, YA_te, YB_te = prepare_dataset(test_exs, channels, lowcut, highcut)
        
        model = EEGNet(in_channels=len(channels)).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = PearsonMSELoss(alpha=0.1)
        
        best_val_acc = 0.0
        best_weights = deepcopy(model.state_dict())
        patience = 15
        epochs_no_improve = 0
        
        for epoch in range(100):
            model.train()
            for i in range(len(X_tr)):
                x = torch.FloatTensor(X_tr[i]).unsqueeze(0).to(device)
                y = torch.FloatTensor(Y_tr[i]).unsqueeze(0).unsqueeze(0).to(device)
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
            
            nc_va, nt_va = evaluate_model(model, X_va, YA_va, YB_va, device)
            val_acc = nc_va / max(nt_va, 1)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
        
        model.load_state_dict(best_weights)
        nc_te, nt_te = evaluate_model(model, X_te, YA_te, YB_te, device)
        acc = nc_te / max(nt_te, 1)
        all_accs.append(acc)
    
    mean_acc = np.mean(all_accs)
    return mean_acc

def main():
    parser = argparse.ArgumentParser(description="EEGNet-Specific Channel Discovery")
    parser.add_argument("--lowcut", type=float, default=1.0)
    parser.add_argument("--highcut", type=float, default=6.0)
    parser.add_argument("--num-eeg-channels", type=int, default=64,
                        help="Total number of EEG channels in the dataset (default: 64)")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Band: {args.lowcut}-{args.highcut} Hz")
    print(f"Total channels to rank: {args.num_eeg_channels}")
    
    all_paths = subject_files()
    if not all_paths:
        print("No subjects found.")
        return
    
    paths = [p for p in all_paths if p.stem in SCREENING_SUBJECTS]
    print(f"Screening subjects: {[p.stem for p in paths]}")
    
    subject_examples = {str(p): load_subject_examples(p) for p in paths}
    
    # Determine how many EEG channels are available
    sample_ex = list(subject_examples.values())[0][0]
    total_ch = sample_ex.eeg.shape[1]
    print(f"Detected {total_ch} channels in data")
    all_channels = list(range(min(args.num_eeg_channels, total_ch)))
    
    # ============================================================
    # PHASE 1: Train 64-channel EEGNet and extract channel rankings
    # ============================================================
    print("\n" + "=" * 60)
    print("PHASE 1: Training 64-channel EEGNet for channel ranking")
    print("=" * 60)
    
    importance = train_and_extract(paths, subject_examples, all_channels, 
                                    args.lowcut, args.highcut, device)
    
    # Rank channels by importance (descending)
    ranked_indices = np.argsort(importance)[::-1]
    
    print("\n" + "=" * 60)
    print("CHANNEL IMPORTANCE RANKING (EEGNet Spatial Filters)")
    print("=" * 60)
    print(f"{'Rank':<6} {'Index':<8} {'Name':<10} {'Importance':<12} {'In Current?'}")
    print("-" * 50)
    for rank, idx in enumerate(ranked_indices):
        name = DTU_CHANNEL_NAMES[idx] if idx < len(DTU_CHANNEL_NAMES) else f"Ch{idx}"
        in_current = "  ✓" if idx in CURRENT_BEST_CHANNELS else ""
        print(f"{rank+1:<6} {idx:<8} {name:<10} {importance[idx]:<12.6f} {in_current}")
    
    # Generate channel sets
    top4 = ranked_indices[:4].tolist()
    top8 = ranked_indices[:8].tolist()
    top12 = ranked_indices[:12].tolist()
    top16 = ranked_indices[:16].tolist()
    
    print(f"\nTop  4: {top4} = {[DTU_CHANNEL_NAMES[i] if i < len(DTU_CHANNEL_NAMES) else f'Ch{i}' for i in top4]}")
    print(f"Top  8: {top8} = {[DTU_CHANNEL_NAMES[i] if i < len(DTU_CHANNEL_NAMES) else f'Ch{i}' for i in top8]}")
    print(f"Top 12: {top12} = {[DTU_CHANNEL_NAMES[i] if i < len(DTU_CHANNEL_NAMES) else f'Ch{i}' for i in top12]}")
    print(f"Top 16: {top16} = {[DTU_CHANNEL_NAMES[i] if i < len(DTU_CHANNEL_NAMES) else f'Ch{i}' for i in top16]}")
    
    # Overlap with current best
    overlap = set(top8) & set(CURRENT_BEST_CHANNELS)
    print(f"\nOverlap between EEGNet Top8 and Ridge Top8: {len(overlap)}/8 channels")
    print(f"  Shared: {[DTU_CHANNEL_NAMES[i] if i < len(DTU_CHANNEL_NAMES) else f'Ch{i}' for i in overlap]}")
    
    # ============================================================
    # PHASE 2: Screen each channel set
    # ============================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Screening channel sets (4-subject LOSO)")
    print("=" * 60)
    
    configs = [
        ("Current Ridge 8ch", CURRENT_BEST_CHANNELS),
        ("EEGNet Top 4",  top4),
        ("EEGNet Top 8",  top8),
        ("EEGNet Top 12", top12),
        ("EEGNet Top 16", top16),
    ]
    
    results = []
    for name, channels in configs:
        print(f"\n  Screening: {name} ({len(channels)} channels: {channels})")
        acc = screen_channel_set(name, channels, paths, subject_examples, 
                                  args.lowcut, args.highcut, device)
        print(f"  => {name}: {acc*100:.2f}%")
        results.append((name, channels, acc))
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 60)
    print("EEGNET CHANNEL DISCOVERY RESULTS")
    print("=" * 60)
    print(f"{'Config':<25} {'Channels':<5} {'Accuracy':<10} {'vs Current'}")
    print("-" * 55)
    baseline_acc = results[0][2]
    for name, channels, acc in results:
        delta = (acc - baseline_acc) * 100
        marker = " ← CURRENT" if name == "Current Ridge 8ch" else ""
        best_marker = " ★ BEST" if acc == max(r[2] for r in results) else ""
        print(f"{name:<25} {len(channels):<5} {acc*100:.2f}%     {delta:+.2f}%{marker}{best_marker}")
    
    best_name, best_channels, best_acc = max(results, key=lambda r: r[2])
    print(f"\nBest configuration: {best_name}")
    print(f"Best channels: {best_channels}")
    print(f"Best accuracy: {best_acc*100:.2f}%")
    
    if best_acc > baseline_acc + 0.01:
        print(f"\n>> VERDICT: IMPROVEMENT FOUND ({(best_acc-baseline_acc)*100:+.2f}%).")
        print(f">> Recommend full 18-subject LOSO with {best_name}.")
    else:
        print(f"\n>> VERDICT: NO MEANINGFUL IMPROVEMENT (<1%).")
        print(f">> Current Ridge-discovered channels remain optimal for EEGNet.")

if __name__ == "__main__":
    main()
