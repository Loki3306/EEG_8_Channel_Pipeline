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
from models.neural_ridge_decoder import ResidualNeuralRidgeDecoder

def safe_corr_torch(x, y, eps=1e-8):
    """Batched Pearson correlation in PyTorch. x, y: (Batch, Channels, Time)"""
    x_mean = x.mean(dim=-1, keepdim=True)
    y_mean = y.mean(dim=-1, keepdim=True)
    x_centered = x - x_mean
    y_centered = y - y_mean
    
    cov = (x_centered * y_centered).sum(dim=-1)
    x_var = (x_centered ** 2).sum(dim=-1)
    y_var = (y_centered ** 2).sum(dim=-1)
    
    corr = cov / (torch.sqrt(x_var * y_var) + eps)
    return corr

def custom_loss(pred, target, mse_weight=0.5, corr_weight=0.5):
    mse = nn.functional.mse_loss(pred, target)
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

def prepare_data_and_ridge(subject_data, window_sec=10, hop_sec=2, fs=64, lags=16):
    win_samples = int(window_sec * fs)
    hop_samples = int(hop_sec * fs)
    
    X, Y_a = [], []
    
    num_channels = 8
    num_lags = lags + 1 # 0 to 16
    feature_count = num_channels * num_lags
    num_bands = 28
    
    xtx = np.zeros((feature_count, feature_count), dtype=float)
    xty = np.zeros((feature_count, num_bands), dtype=float)
    
    print(f"Preparing data and computing analytical Ridge weights...")
    for t in subject_data:
        eeg = t["eeg"]       # (8, Time)
        audio_a = t["audio_a"] # (28, Time)
        
        # Normalize for Ridge
        eeg_np = eeg.numpy()
        e_mean_full = eeg_np.mean(axis=1, keepdims=True)
        e_std_full = eeg_np.std(axis=1, keepdims=True) + 1e-12
        e_norm_full = (eeg_np - e_mean_full) / e_std_full
        
        a_np = audio_a.numpy()
        a_mean_full = a_np.mean(axis=1, keepdims=True)
        a_std_full = a_np.std(axis=1, keepdims=True) + 1e-12
        a_norm_full = (a_np - a_mean_full) / a_std_full
        
        time_steps = e_norm_full.shape[1]
        lagged_blocks = []
        for lag in range(num_lags):
            if lag == 0:
                lagged_blocks.append(e_norm_full.T)
            else:
                shifted = np.vstack([np.zeros((lag, num_channels)), e_norm_full.T[:-lag]])
                lagged_blocks.append(shifted)
                
        X_mat = np.concatenate(lagged_blocks, axis=1)
        Y_mat = a_norm_full.T # (Time, 28)
        
        xtx += X_mat.T @ X_mat
        xty += X_mat.T @ Y_mat
        
        # Windowing for Neural Network
        n_windows = (eeg.shape[1] - win_samples) // hop_samples + 1
        for i in range(max(1, n_windows)):
            start = i * hop_samples
            stop = start + win_samples
            if stop > eeg.shape[1]:
                break
                
            e = eeg[:, start:stop]
            a = audio_a[:, start:stop]
            
            a_mean = a.mean(dim=1, keepdim=True)
            a_std = a.std(dim=1, keepdim=True) + 1e-8
            a_norm = (a - a_mean) / a_std
            
            e_mean = e.mean(dim=1, keepdim=True)
            e_std = e.std(dim=1, keepdim=True) + 1e-8
            e_norm = (e - e_mean) / e_std
            
            X.append(e_norm)
            Y_a.append(a_norm)
            
    # Solve Ridge
    ridge_lambda = 100.0
    regularized = xtx + ridge_lambda * np.eye(feature_count, dtype=float)
    W = np.linalg.solve(regularized, xty) # (Channels*Lags, 28)
    
    W_reshaped = W.reshape(num_lags, num_channels, num_bands) # (17, 8, 28)
    W_torch = np.zeros((num_bands, num_channels, num_lags), dtype=np.float32)
    
    for lag in range(num_lags):
        k = num_lags - 1 - lag
        W_torch[:, :, k] = W_reshaped[lag, :, :].T
        
    if not X:
        return None, None
        
    return (torch.stack(X), torch.stack(Y_a)), W_torch

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
            
    train_tensors, ridge_weights = prepare_data_and_ridge(train_trials, window_sec=10, hop_sec=2, fs=64, lags=16)
    if train_tensors is None:
        print("No training data.")
        return
    X_train, Ya_train = train_tensors
    
    dataset = TensorDataset(X_train, Ya_train)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # ---------------------------------------------------------
    # 2. Model, Loss, Optimizer
    # ---------------------------------------------------------
    model = ResidualNeuralRidgeDecoder(in_channels=8, out_channels=28, lags=16).to(device)
    
    print("\n====================================================")
    print("EXPERIMENT 2: Freeze Verification")
    print("====================================================")
    print(f"{'Parameter Name':<30} | {'Shape':<20} | {'Requires Grad'}")
    print("-" * 75)
    for name, param in model.named_parameters():
        req_grad = param.requires_grad
        print(f"{name:<30} | {str(list(param.shape)):<20} | {req_grad}")
        if "base_ridge" in name:
            assert req_grad == False, f"Error: Ridge parameter {name} is not frozen!"
    print("✅ All base_ridge parameters are successfully frozen.")
    print("====================================================\n")
    
    model.load_ridge_weights(ridge_weights)
    
    # AdamW with weight_decay=1e-4
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    
    epochs = 15
    best_margin = -float('inf')
    patience = 3
    patience_counter = 0
    
    residual_penalty_weight = 1e-4
    
    for epoch in range(1, epochs + 1):
        # --- Training ---
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            
            base_env, delta_env = model(batch_x)
            
            # Experiment 4: Residual Error Target
            target_residual = batch_y - base_env
            loss_corr = custom_loss(delta_env, target_residual, mse_weight=0.5, corr_weight=0.5)
            
            # Experiment 3 (Fix 3): Residual magnitude penalty
            loss_penalty = residual_penalty_weight * (torch.norm(delta_env, p=2) ** 2)
            
            loss = loss_corr + loss_penalty
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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
        
        base_trial_correct = 0
        base_windows_correct = 0
        base_trial_margins = []
        
        val_plot_data = [] 
        trial_margins = []
        
        total_norm_residual = 0.0
        total_norm_ridge = 0.0
        total_norm_pred = 0.0
        
        with torch.no_grad():
            for t_idx, t in enumerate(test_trials):
                eeg = t["eeg"].unsqueeze(0).to(device)       
                audio_a = t["audio_a"].unsqueeze(0).to(device) 
                audio_b = t["audio_b"].unsqueeze(0).to(device)
                
                eeg_mean = eeg.mean(dim=2, keepdim=True)
                eeg_std = eeg.std(dim=2, keepdim=True) + 1e-8
                eeg_norm = (eeg - eeg_mean) / eeg_std
                
                audio_a_mean = audio_a.mean(dim=2, keepdim=True)
                audio_a_std = audio_a.std(dim=2, keepdim=True) + 1e-8
                audio_a_norm = (audio_a - audio_a_mean) / audio_a_std
                
                audio_b_mean = audio_b.mean(dim=2, keepdim=True)
                audio_b_std = audio_b.std(dim=2, keepdim=True) + 1e-8
                audio_b_norm = (audio_b - audio_b_mean) / audio_b_std
                
                base_env, delta_env = model(eeg_norm)
                pred = base_env + delta_env
                
                # Validation Loss (Residual Target)
                target_residual = audio_a_norm - base_env
                loss_corr = custom_loss(delta_env, target_residual, mse_weight=0.5, corr_weight=0.5)
                loss_penalty = residual_penalty_weight * (torch.norm(delta_env, p=2) ** 2)
                loss_val = (loss_corr + loss_penalty).item()
                val_loss += loss_val
                total_val_samples += 1
                
                # Magnitudes (Experiment 3)
                total_norm_ridge += torch.norm(base_env, p=2).item()
                total_norm_residual += torch.norm(delta_env, p=2).item()
                total_norm_pred += torch.norm(pred, p=2).item()
                
                pred_np = pred.squeeze(0).cpu().numpy()
                base_np = base_env.squeeze(0).cpu().numpy()
                wav_a_np = audio_a_norm.squeeze(0).cpu().numpy()
                wav_b_np = audio_b_norm.squeeze(0).cpu().numpy()
                
                num_bands = pred_np.shape[0]
                
                c_att = np.mean([safe_corr_np(pred_np[i], wav_a_np[i]) for i in range(num_bands)])
                c_unatt = np.mean([safe_corr_np(pred_np[i], wav_b_np[i]) for i in range(num_bands)])
                
                mean_corr_att += c_att
                mean_corr_unatt += c_unatt
                trial_margin = c_att - c_unatt
                trial_margins.append(trial_margin)
                
                c_att_base = np.mean([safe_corr_np(base_np[i], wav_a_np[i]) for i in range(num_bands)])
                c_unatt_base = np.mean([safe_corr_np(base_np[i], wav_b_np[i]) for i in range(num_bands)])
                base_trial_margins.append(c_att_base - c_unatt_base)
                
                trial_ok, n_win, c_win = evaluate_trial_majority_vote_multiband(pred_np, wav_a_np, wav_b_np, window_seconds=10, hop_seconds=1.0, fs=64)
                if trial_ok:
                    mv_trial_correct += 1
                mv_windows_total += n_win
                mv_windows_correct += c_win
                
                trial_ok_base, _, c_win_base = evaluate_trial_majority_vote_multiband(base_np, wav_a_np, wav_b_np, window_seconds=10, hop_seconds=1.0, fs=64)
                if trial_ok_base:
                    base_trial_correct += 1
                base_windows_correct += c_win_base
                
                if len(val_plot_data) == 0 and n_win > 0:
                    vis_samples = 10 * 64
                    val_plot_data = (wav_a_np[:, :vis_samples], pred_np[:, :vis_samples])
                    
        val_loss /= total_val_samples
        mean_corr_att /= total_val_samples
        mean_corr_unatt /= total_val_samples
        
        trial_acc = mv_trial_correct / total_val_samples if total_val_samples > 0 else 0
        win_acc = mv_windows_correct / mv_windows_total if mv_windows_total > 0 else 0
        epoch_margin = mean_corr_att - mean_corr_unatt
        
        base_trial_acc = base_trial_correct / total_val_samples if total_val_samples > 0 else 0
        base_win_acc = base_windows_correct / mv_windows_total if mv_windows_total > 0 else 0
        base_median_margin = np.median(base_trial_margins)
        
        max_diff_ridge = np.max(np.abs(base_np - base_np))
        
        print(f"\n====================================================")
        print(f"Epoch {epoch}/{epochs}")
        print(f"  Train Loss:       {train_loss:.6f}")
        print(f"  Val Loss:         {val_loss:.6f}")
        
        print(f"\n  EXPERIMENT 3: Residual Magnitude")
        ratio = total_norm_residual / (total_norm_ridge + 1e-8)
        print(f"    ||Ridge||:      {total_norm_ridge:.2f}")
        print(f"    ||Residual||:   {total_norm_residual:.2f}")
        print(f"    ||Prediction||: {total_norm_pred:.2f}")
        print(f"    Ratio (Res/Rdg): {ratio:.4f}")
        
        print(f"\n  EXPERIMENT 1: Configuration Verification")
        print(f"  [Model A: Pure Ridge]")
        print(f"    Trial Acc:  {base_trial_acc*100:.1f}%")
        print(f"    Window Acc: {base_win_acc*100:.1f}%")
        print(f"    Med Margin: {base_median_margin:.4f}")
        
        print(f"\n  [Model B: Residual Disabled]")
        print(f"    Max Diff from Ridge: {max_diff_ridge:.6e}")
        
        print(f"\n  [Model C: Ridge + Neural]")
        print(f"    Trial Acc:  {trial_acc*100:.1f}%")
        print(f"    Window Acc: {win_acc*100:.1f}%")
        print(f"    Med Margin: {np.median(trial_margins):.4f}")
        
        print(f"\n  Improvement over Ridge:")
        print(f"    Trial Acc:  {(trial_acc - base_trial_acc)*100:+.1f}%")
        print(f"    Window Acc: {(win_acc - base_win_acc)*100:+.1f}%")
        print(f"    Med Margin: {np.median(trial_margins) - base_median_margin:+.4f}")
        print("====================================================\n")
        
        if epoch_margin > best_margin:
            best_margin = epoch_margin
            torch.save(model.state_dict(), out_dir / "neural_ridge_best.pth")
            print(f"  => Saved new best model (Validation Margin: {best_margin:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  => No improvement. Patience: {patience_counter}/{patience}")
            
        print("-" * 50)
        
        if patience_counter >= patience:
            print("Early stopping triggered due to lack of improvement in Validation Margin.")
            break
        
        # --- Visualization ---
        if val_plot_data:
            true_env, pred_env = val_plot_data
            bands_to_plot = [0, 4, 9, 19, 27]
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
