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
from training.loso_ridge_runner import evaluate_trial_windows
from evaluation.aad_metrics import safe_corr

def evaluate_trial_majority_vote(predicted: np.ndarray, wav_a: np.ndarray, wav_b: np.ndarray, window_seconds: int, fs: int = 64):
    if window_seconds <= 0:
        corr_a = safe_corr(predicted, wav_a)
        corr_b = safe_corr(predicted, wav_b)
        return corr_a > corr_b, 1, 1 if corr_a > corr_b else 0
        
    window_samples = window_seconds * fs
    if window_samples >= predicted.size:
        corr_a = safe_corr(predicted, wav_a)
        corr_b = safe_corr(predicted, wav_b)
        return corr_a > corr_b, 1, 1 if corr_a > corr_b else 0
        
    correct_windows = 0
    total_windows = 0
    
    for start in range(0, predicted.size - window_samples + 1, window_samples):
        stop = start + window_samples
        corr_a = safe_corr(predicted[start:stop], wav_a[start:stop])
        corr_b = safe_corr(predicted[start:stop], wav_b[start:stop])
        if corr_a > corr_b:
            correct_windows += 1
        total_windows += 1
        
    if total_windows == 0:
        return False, 0, 0
        
    trial_correct = (correct_windows > total_windows / 2.0)
    return trial_correct, total_windows, correct_windows

def prepare_data(subject_data, window_sec=2, fs=64):
    """Slices trials into fixed windows for mini-batch training."""
    win_samples = window_sec * fs
    
    X, Y_a, Y_b = [], [], []
    for t in subject_data:
        eeg = t["eeg"]       # (8, Time)
        audio_a = t["audio_a"] # (28, Time)
        audio_b = t["audio_b"]
        
        # Keep full trials for evaluation later, but slice for training
        n_windows = eeg.shape[1] // win_samples
        for i in range(n_windows):
            start = i * win_samples
            stop = start + win_samples
            X.append(eeg[:, start:stop])
            Y_a.append(audio_a[:, start:stop])
            Y_b.append(audio_b[:, start:stop])
            
    if not X:
        return None
        
    return torch.stack(X), torch.stack(Y_a), torch.stack(Y_b)

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
    
    # Setup directories
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
            
    # Prepare chunked training data
    train_tensors = prepare_data(train_trials, window_sec=2, fs=64)
    if train_tensors is None:
        print("No training data.")
        return
    X_train, Ya_train, Yb_train = train_tensors
    
    dataset = TensorDataset(X_train, Ya_train)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # ---------------------------------------------------------
    # 2. Model, Loss, Optimizer
    # ---------------------------------------------------------
    model = NeuralRidgeDecoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 5
    
    for epoch in range(1, epochs + 1):
        # --- Training ---
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            pred = model(batch_x)
            
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            
        train_loss /= len(dataset)
        
        # --- Evaluation ---
        model.eval()
        
        val_mse = 0.0
        total_val_samples = 0
        
        mean_corr_att = 0.0
        mean_corr_unatt = 0.0
        
        mv_trial_correct = 0
        mv_windows_correct = 0
        mv_windows_total = 0
        
        val_plot_data = [] # For visualization
        
        with torch.no_grad():
            for t in test_trials:
                eeg = t["eeg"].unsqueeze(0).to(device)       # (1, 8, Time)
                audio_a = t["audio_a"].unsqueeze(0).to(device) # (1, 28, Time)
                audio_b = t["audio_b"].unsqueeze(0).to(device)
                
                pred = model(eeg) # (1, 28, Time)
                
                # Validation MSE over full trial
                mse = criterion(pred, audio_a).item()
                val_mse += mse
                total_val_samples += 1
                
                # To match Ridge perfectly, average the 28 channels down to 1 envelope
                pred_1d = pred.squeeze(0).mean(dim=0).cpu().numpy()
                wav_a_1d = audio_a.squeeze(0).mean(dim=0).cpu().numpy()
                wav_b_1d = audio_b.squeeze(0).mean(dim=0).cpu().numpy()
                
                # Compute full-trial correlations
                c_att = safe_corr(pred_1d, wav_a_1d)
                c_unatt = safe_corr(pred_1d, wav_b_1d)
                
                mean_corr_att += c_att
                mean_corr_unatt += c_unatt
                
                # AAD Window Evaluation
                trial_ok, n_win, c_win = evaluate_trial_majority_vote(pred_1d, wav_a_1d, wav_b_1d, window_seconds=2, fs=64)
                if trial_ok:
                    mv_trial_correct += 1
                mv_windows_total += n_win
                mv_windows_correct += c_win
                
                # Save plot data for the first trial (first few windows)
                if len(val_plot_data) < 5 and n_win > 0:
                    val_plot_data.append((wav_a_1d[:128], pred_1d[:128]))
                    
        val_mse /= total_val_samples
        mean_corr_att /= total_val_samples
        mean_corr_unatt /= total_val_samples
        
        trial_acc = mv_trial_correct / total_val_samples if total_val_samples > 0 else 0
        win_acc = mv_windows_correct / mv_windows_total if mv_windows_total > 0 else 0
        
        print(f"Epoch {epoch}/{epochs}")
        print(f"  Train MSE Loss: {train_loss:.6f}")
        print(f"  Val MSE Loss:   {val_mse:.6f}")
        print(f"  Mean Corr Att:  {mean_corr_att:.4f}")
        print(f"  Mean Corr Unatt:{mean_corr_unatt:.4f}")
        print(f"  Window Acc:     {win_acc*100:.1f}%")
        print(f"  Trial Acc:      {trial_acc*100:.1f}%")
        print("-" * 50)
        
        # --- Visualization ---
        plt.figure(figsize=(15, 8))
        for i, (true_env, pred_env) in enumerate(val_plot_data):
            plt.subplot(5, 1, i+1)
            # Normalize for visualization comparison
            true_env = (true_env - true_env.mean()) / (true_env.std() + 1e-8)
            pred_env = (pred_env - pred_env.mean()) / (pred_env.std() + 1e-8)
            
            plt.plot(true_env, label="True Attended", alpha=0.7)
            plt.plot(pred_env, label="Predicted", alpha=0.7)
            if i == 0:
                plt.title(f"Epoch {epoch}: True vs Predicted Envelope (First 2s)")
            if i == 4:
                plt.xlabel("Samples")
            plt.legend(loc="upper right")
            
        plt.tight_layout()
        plt.savefig(out_dir / f"reconstruction_epoch_{epoch}.png")
        plt.close()

if __name__ == "__main__":
    main()
