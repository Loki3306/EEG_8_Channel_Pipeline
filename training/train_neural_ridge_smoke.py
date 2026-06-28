import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.kul_cached_dataset import KULCachedLoader
from models.neural_ridge_decoder import NeuralRidgeDecoder

def safe_corr_torch(x, y, eps=1e-8):
    """Batched Pearson correlation in PyTorch. x, y: (Batch, Channels, Time)"""
    x_mean = x.mean(dim=-1, keepdim=True)
    y_mean = y.mean(dim=-1, keepdim=True)
    x_centered = x - x_mean
    y_centered = y - y_mean
    
    cov = (x_centered * y_centered).sum(dim=-1)
    x_var = (x_centered ** 2).sum(dim=-1)
    y_var = (y_centered ** 2).sum(dim=-1)
    
    # corr: (Batch, Channels)
    corr = cov / (torch.sqrt(x_var * y_var) + eps)
    return corr

def custom_loss(pred, target, mse_weight=0.5, corr_weight=0.5):
    mse = nn.functional.mse_loss(pred, target)
    # Compute correlation per channel and average across channels and batch
    corr = safe_corr_torch(pred, target)
    mean_corr = corr.mean()
    corr_loss = 1.0 - mean_corr
    return mse_weight * mse + corr_weight * corr_loss

def safe_corr_np(x, y, eps=1e-8):
    x_mean = x.mean()
    y_mean = y.mean()
    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
    return num / (den + eps)

def evaluate_trial_majority_vote_multiband(predicted: np.ndarray, wav_a: np.ndarray, wav_b: np.ndarray, window_seconds: int, hop_seconds: float = 1.0, fs: int = 64):
    """
    Evaluates correlation across all bands over overlapping windows.
    predicted, wav_a, wav_b: shape (Channels, Time)
    """
    num_bands = predicted.shape[0]
    win_samples = int(window_seconds * fs)
    hop_samples = int(hop_seconds * fs)
    
    if win_samples >= predicted.shape[1]:
        c_a = np.mean([safe_corr_np(predicted[i], wav_a[i]) for i in range(num_bands)])
        c_b = np.mean([safe_corr_np(predicted[i], wav_b[i]) for i in range(num_bands)])
        return c_a > c_b, 1, 1 if c_a > c_b else 0
        
    correct_windows = 0
    total_windows = 0
    
    for start in range(0, predicted.shape[1] - win_samples + 1, hop_samples):
        stop = start + win_samples
        c_a = np.mean([safe_corr_np(predicted[i, start:stop], wav_a[i, start:stop]) for i in range(num_bands)])
        c_b = np.mean([safe_corr_np(predicted[i, start:stop], wav_b[i, start:stop]) for i in range(num_bands)])
        if c_a > c_b:
            correct_windows += 1
        total_windows += 1
        
    if total_windows == 0:
        return False, 0, 0
        
    trial_correct = (correct_windows > total_windows / 2.0)
    return trial_correct, total_windows, correct_windows

def prepare_data(subject_data, window_sec=10, hop_sec=2, fs=64):
    """Slices trials into fixed windows for mini-batch training."""
    win_samples = int(window_sec * fs)
    hop_samples = int(hop_sec * fs)
    
    X, Y_a = [], []
    for t in subject_data:
        eeg = t["eeg"]       # (8, Time)
        audio_a = t["audio_a"] # (28, Time)
        
        n_windows = (eeg.shape[1] - win_samples) // hop_samples + 1
        for i in range(max(1, n_windows)):
            start = i * hop_samples
            stop = start + win_samples
            if stop > eeg.shape[1]:
                break
                
            e = eeg[:, start:stop]
            a = audio_a[:, start:stop]
            
            # Normalize target (attended envelope) to mean 0, std 1 per band
            a_mean = a.mean(dim=1, keepdim=True)
            a_std = a.std(dim=1, keepdim=True) + 1e-8
            a_norm = (a - a_mean) / a_std
            
            # Normalize EEG per channel per window
            e_mean = e.mean(dim=1, keepdim=True)
            e_std = e.std(dim=1, keepdim=True) + 1e-8
            e_norm = (e - e_mean) / e_std
            
            X.append(e_norm)
            Y_a.append(a_norm)
            
    if not X:
        return None
        
    return torch.stack(X), torch.stack(Y_a)

def main():
    print("Loading KUL Cache...")
    loader = KULCachedLoader(REPO_ROOT / "data" / "processed_kul")
    try:
        all_subject_data = loader.load_all()
    except FileNotFoundError:
        print("KUL Cache not found.")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    out_dir = REPO_ROOT / "results" / "neural_ridge_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # ---------------------------------------------------------
    # 1. Data Split (S1 Test, S2-S16 Train)
    # ---------------------------------------------------------
    test_subject = "S1"
    train_trials = []
    test_trials = all_subject_data[test_subject]
    
    for sub, trials in all_subject_data.items():
        if sub != test_subject:
            train_trials.extend(trials)
            
    # Prepare chunked training data (10s windows, 2s hop for dense training)
    train_tensors = prepare_data(train_trials, window_sec=10, hop_sec=2, fs=64)
    if train_tensors is None:
        print("No training data.")
        return
    X_train, Ya_train = train_tensors
    
    dataset = TensorDataset(X_train, Ya_train)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # ---------------------------------------------------------
    # 2. Model, Loss, Optimizer
    # ---------------------------------------------------------
    model = NeuralRidgeDecoder(in_channels=8, out_channels=28).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 5
    best_margin = -float('inf')
    
    for epoch in range(1, epochs + 1):
        # --- Training ---
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            pred = model(batch_x)
            
            # Loss = 0.5 * MSE + 0.5 * (1 - Pearson)
            loss = custom_loss(pred, batch_y, mse_weight=0.5, corr_weight=0.5)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            
        train_loss /= len(dataset)
        
        # --- Evaluation ---
        model.eval()
        
        val_loss = 0.0
        total_val_samples = 0
        
        mean_corr_att = 0.0
        mean_corr_unatt = 0.0
        
        mv_trial_correct = 0
        mv_windows_correct = 0
        mv_windows_total = 0
        
        val_plot_data = [] # For visualization
        trial_margins = []
        
        with torch.no_grad():
            for t_idx, t in enumerate(test_trials):
                eeg = t["eeg"].unsqueeze(0).to(device)       # (1, 8, Time)
                audio_a = t["audio_a"].unsqueeze(0).to(device) # (1, 28, Time)
                audio_b = t["audio_b"].unsqueeze(0).to(device)
                
                # Window-level normalization for full test trial EEG
                eeg_mean = eeg.mean(dim=2, keepdim=True)
                eeg_std = eeg.std(dim=2, keepdim=True) + 1e-8
                eeg_norm = (eeg - eeg_mean) / eeg_std
                
                # Normalize targets identically to training
                audio_a_mean = audio_a.mean(dim=2, keepdim=True)
                audio_a_std = audio_a.std(dim=2, keepdim=True) + 1e-8
                audio_a_norm = (audio_a - audio_a_mean) / audio_a_std
                
                audio_b_mean = audio_b.mean(dim=2, keepdim=True)
                audio_b_std = audio_b.std(dim=2, keepdim=True) + 1e-8
                audio_b_norm = (audio_b - audio_b_mean) / audio_b_std
                
                pred = model(eeg_norm) # (1, 28, Time)
                
                # Eval Loss
                loss_val = custom_loss(pred, audio_a_norm, mse_weight=0.5, corr_weight=0.5).item()
                val_loss += loss_val
                total_val_samples += 1
                
                pred_np = pred.squeeze(0).cpu().numpy()
                wav_a_np = audio_a_norm.squeeze(0).cpu().numpy()
                wav_b_np = audio_b_norm.squeeze(0).cpu().numpy()
                
                num_bands = pred_np.shape[0]
                
                # Compute full-trial mean correlations across all bands
                c_att = np.mean([safe_corr_np(pred_np[i], wav_a_np[i]) for i in range(num_bands)])
                c_unatt = np.mean([safe_corr_np(pred_np[i], wav_b_np[i]) for i in range(num_bands)])
                
                mean_corr_att += c_att
                mean_corr_unatt += c_unatt
                
                trial_margin = c_att - c_unatt
                trial_margins.append(trial_margin)
                
                # AAD Window Evaluation (10s windows, 1s hop)
                trial_ok, n_win, c_win = evaluate_trial_majority_vote_multiband(pred_np, wav_a_np, wav_b_np, window_seconds=10, hop_seconds=1.0, fs=64)
                if trial_ok:
                    mv_trial_correct += 1
                mv_windows_total += n_win
                mv_windows_correct += c_win
                
                # Save plot data for the first trial
                if len(val_plot_data) == 0 and n_win > 0:
                    vis_samples = 10 * 64
                    val_plot_data = (wav_a_np[:, :vis_samples], pred_np[:, :vis_samples])
                    
        val_loss /= total_val_samples
        mean_corr_att /= total_val_samples
        mean_corr_unatt /= total_val_samples
        
        trial_acc = mv_trial_correct / total_val_samples if total_val_samples > 0 else 0
        win_acc = mv_windows_correct / mv_windows_total if mv_windows_total > 0 else 0
        epoch_margin = mean_corr_att - mean_corr_unatt
        
        print(f"Epoch {epoch}/{epochs}")
        print(f"  Train Loss:       {train_loss:.6f}")
        print(f"  Val Loss:         {val_loss:.6f}")
        print(f"  Mean Corr(att):   {mean_corr_att:.4f}")
        print(f"  Mean Corr(unatt): {mean_corr_unatt:.4f}")
        print(f"  Overall Margin:   {epoch_margin:.4f}")
        
        print(f"\n  [Trial Margins]")
        for i, m in enumerate(trial_margins):
            print(f"    Trial {i+1:02d}: {m:.4f}")
            
        print(f"\n  [Margin Stats]")
        print(f"    Median: {np.median(trial_margins):.4f}")
        print(f"    Std:    {np.std(trial_margins):.4f}")
        print(f"    Min:    {np.min(trial_margins):.4f}")
        print(f"    Max:    {np.max(trial_margins):.4f}")
        
        print(f"\n  Window Acc:       {win_acc*100:.1f}%")
        print(f"  Trial Acc:        {trial_acc*100:.1f}%")
        print("-" * 50)
        
        # Save best checkpoint
        if epoch_margin > best_margin:
            best_margin = epoch_margin
            torch.save(model.state_dict(), out_dir / "neural_ridge_best.pth")
            print(f"  => Saved new best model (Margin: {best_margin:.4f})")
            print("-" * 50)
        
        # --- Visualization ---
        if val_plot_data:
            true_env, pred_env = val_plot_data
            bands_to_plot = [0, 4, 9, 19, 27]
            
            # Ensure we don't try to plot indices out of bounds if num_bands changed
            bands_to_plot = [b for b in bands_to_plot if b < true_env.shape[0]]
            
            plt.figure(figsize=(15, 10))
            for i, band_idx in enumerate(bands_to_plot):
                plt.subplot(len(bands_to_plot), 1, i+1)
                
                t_env = true_env[band_idx]
                p_env = pred_env[band_idx]
                
                t_env = (t_env - t_env.mean()) / (t_env.std() + 1e-8)
                p_env = (p_env - p_env.mean()) / (p_env.std() + 1e-8)
                
                plt.plot(t_env, label=f"True Attended (Band {band_idx+1})", alpha=0.7)
                plt.plot(p_env, label=f"Predicted (Band {band_idx+1})", alpha=0.7)
                
                if i == 0:
                    plt.title(f"Epoch {epoch}: True vs Predicted Envelope (First 10s)")
                if i == len(bands_to_plot) - 1:
                    plt.xlabel("Samples")
                plt.legend(loc="upper right")
                
            plt.tight_layout()
            plt.savefig(out_dir / f"reconstruction_epoch_{epoch}.png")
            plt.close()

if __name__ == "__main__":
    main()
