import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import json

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.kul_cached_dataset import KULCachedLoader
from models.aad_conformer import AADConformer

def safe_corr_torch(x, y, eps=1e-8):
    """Batched Pearson correlation in PyTorch. x, y: (Batch, Time)"""
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

def evaluate_trial_majority_vote(predicted: np.ndarray, wav_a: np.ndarray, wav_b: np.ndarray, window_seconds: int, hop_seconds: float = 1.0, fs: int = 64):
    """Majority vote for a single-band predicted envelope."""
    win_samples = int(window_seconds * fs)
    hop_samples = int(hop_seconds * fs)
    
    if win_samples >= predicted.shape[0]:
        c_a = safe_corr_np(predicted, wav_a)
        c_b = safe_corr_np(predicted, wav_b)
        return c_a > c_b, 1, 1 if c_a > c_b else 0
        
    correct_windows = 0
    total_windows = 0
    
    for start in range(0, predicted.shape[0] - win_samples + 1, hop_samples):
        stop = start + win_samples
        c_a = safe_corr_np(predicted[start:stop], wav_a[start:stop])
        c_b = safe_corr_np(predicted[start:stop], wav_b[start:stop])
        if c_a > c_b:
            correct_windows += 1
        total_windows += 1
        
    if total_windows == 0:
        return False, 0, 0
        
    trial_correct = (correct_windows > total_windows / 2.0)
    return trial_correct, total_windows, correct_windows

def prepare_data(subject_data, window_sec=10, hop_sec=2, fs=64):
    win_samples = int(window_sec * fs)
    hop_samples = int(hop_sec * fs)
    
    X, Y_a = [], []
    
    for t in subject_data:
        eeg = t["eeg"]       # (8, Time)
        audio_a = t["audio_a"] # (28, Time)
        
        # Audio preprocessing: mean over the 28 subbands to get a single broad envelope
        audio_a = audio_a.mean(dim=0, keepdim=True) # (1, Time)
        
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
            Y_a.append(a_norm.squeeze(0)) # Squeeze to [Time]
            
    if not X:
        return None
        
    return torch.stack(X), torch.stack(Y_a)

def main():
    import random
    
    print("Loading KUL Cache...")
    if Path("/kaggle/input/datasets/lowk1ee/kul-preprocessed-cache/data/processed_kul").exists():
        cache_dir = Path("/kaggle/input/datasets/lowk1ee/kul-preprocessed-cache/data/processed_kul")
    elif Path("/kaggle/input/kul-preprocessed-cache/data/processed_kul").exists():
        cache_dir = Path("/kaggle/input/kul-preprocessed-cache/data/processed_kul")
    else:
        cache_dir = REPO_ROOT / "data" / "processed_kul"
        
    try:
        loader = KULCachedLoader(cache_dir)
        all_subject_data = loader.load_all()
    except FileNotFoundError:
        print(f"KUL Cache not found at {cache_dir}. Cannot run full LOSO.")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    out_dir = REPO_ROOT / "results" / "conformer_loso"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    subjects = sorted(list(all_subject_data.keys()))
    
    SEEDS = [1, 7, 21, 42, 123]
    all_seeds_results = {}
    
    for seed in SEEDS:
        print(f"\n{'#'*60}")
        print(f"STARTING SEED: {seed}")
        print(f"{'#'*60}")
        
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        loso_results = {}
        
        checkpoint_dir = out_dir / "checkpoints" / f"seed_{seed}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        for test_subject in subjects:
            print(f"\n{'='*60}")
            print(f"Starting LOSO Fold: Test Subject = {test_subject} | Seed = {seed}")
            print(f"{'='*60}")
            
            test_trials = all_subject_data[test_subject]
            
            # Pick one subject from the remaining as the validation subject
            remaining_subjects = [s for s in subjects if s != test_subject]
            test_idx = subjects.index(test_subject)
            # Deterministically rotate the validation subject
            val_subject = remaining_subjects[test_idx % len(remaining_subjects)]
            val_trials = all_subject_data[val_subject]
            print(f"Validation Subject: {val_subject} | Training Subjects: {len(remaining_subjects)-1}")
            
            train_trials = []
            for sub in remaining_subjects:
                if sub != val_subject:
                    train_trials.extend(all_subject_data[sub])
                    
            train_tensors = prepare_data(train_trials, window_sec=2, hop_sec=1, fs=64)
            if train_tensors is None:
                print(f"No training data for {test_subject}.")
                continue
                
            X_train, Ya_train = train_tensors
            dataset = TensorDataset(X_train, Ya_train)
            dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
            
            model = AADConformer(
                in_channels=8,
                temporal_filters=32,
                spatial_filters=64,
                embed_dim=64,
                num_heads=4,
                num_layers=2,
                dropout=0.3,
                stride=4
            ).to(device)
            
            optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            
            epochs = 15
            best_val_margin = -float('inf')
            patience = 4
            patience_counter = 0
            
            best_model_state = None
            best_epoch = -1
            
            fold_history = {
                "train_loss": [],
                "val_loss": [],
                "val_margin": []
            }
            
            stopped_epoch = epochs
            
            for epoch in range(1, epochs + 1):
                model.train()
                train_loss = 0.0
                
                for batch_x, batch_y in dataloader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    
                    optimizer.zero_grad()
                    pred = model(batch_x)
                    
                    loss = custom_loss(pred, batch_y, mse_weight=0.5, corr_weight=0.5)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_loss += loss.item() * batch_x.size(0)
                    
                train_loss /= len(dataset)
                
                # --- EVALUATE ON VALIDATION SET ---
                model.eval()
                val_loss = 0.0
                total_val_samples = 0
                mean_corr_att, mean_corr_unatt = 0.0, 0.0
                
                with torch.no_grad():
                    for t_idx, t in enumerate(val_trials):
                        eeg = t["eeg"].unsqueeze(0).to(device)       
                        audio_a = t["audio_a"].unsqueeze(0).to(device) 
                        audio_b = t["audio_b"].unsqueeze(0).to(device)
                        
                        audio_a = audio_a.mean(dim=1, keepdim=True)
                        audio_b = audio_b.mean(dim=1, keepdim=True)
                        
                        eeg_mean = eeg.mean(dim=2, keepdim=True)
                        eeg_std = eeg.std(dim=2, keepdim=True) + 1e-8
                        eeg_norm = (eeg - eeg_mean) / eeg_std
                        
                        audio_a_mean = audio_a.mean(dim=2, keepdim=True)
                        audio_a_std = audio_a.std(dim=2, keepdim=True) + 1e-8
                        audio_a_norm = (audio_a - audio_a_mean) / audio_a_std
                        
                        audio_b_mean = audio_b.mean(dim=2, keepdim=True)
                        audio_b_std = audio_b.std(dim=2, keepdim=True) + 1e-8
                        audio_b_norm = (audio_b - audio_b_mean) / audio_b_std
                        
                        pred = model(eeg_norm) # [1, Time]
                        
                        target = audio_a_norm.squeeze(1) # [1, Time]
                        loss_val = custom_loss(pred, target, mse_weight=0.5, corr_weight=0.5).item()
                        val_loss += loss_val
                        total_val_samples += 1
                        
                        pred_np = pred.squeeze(0).cpu().numpy()
                        wav_a_np = audio_a_norm.squeeze(1).squeeze(0).cpu().numpy()
                        wav_b_np = audio_b_norm.squeeze(1).squeeze(0).cpu().numpy()
                        
                        c_att = safe_corr_np(pred_np, wav_a_np)
                        c_unatt = safe_corr_np(pred_np, wav_b_np)
                        mean_corr_att += c_att
                        mean_corr_unatt += c_unatt
                        
                val_loss /= total_val_samples
                mean_corr_att /= total_val_samples
                mean_corr_unatt /= total_val_samples
                epoch_val_margin = mean_corr_att - mean_corr_unatt
                
                fold_history["train_loss"].append(train_loss)
                fold_history["val_loss"].append(val_loss)
                fold_history["val_margin"].append(epoch_val_margin)
                
                print(f"Fold {test_subject} - Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Margin: {epoch_val_margin:.4f}")
                
                if epoch_val_margin > best_val_margin:
                    best_val_margin = epoch_val_margin
                    best_epoch = epoch
                    import copy
                    best_model_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    print("Early stopping triggered on validation set.")
                    stopped_epoch = epoch
                    break
                    
            # Save checkpoint for Phase 2 (Window Scaling)
            checkpoint_path = checkpoint_dir / f"model_{test_subject}.pt"
            torch.save(best_model_state, checkpoint_path)
            
            # --- EVALUATE ON HELD-OUT TEST SET (ONCE) ---
            print(f"--- Evaluating Test Subject: {test_subject} ---")
            model.load_state_dict(best_model_state)
            model.eval()
            
            test_loss = 0.0
            total_test_samples = 0
            test_corr_att, test_corr_unatt = 0.0, 0.0
            mv_trial_correct, mv_windows_correct, mv_windows_total = 0, 0, 0
            trial_margins = []
            
            with torch.no_grad():
                for t_idx, t in enumerate(test_trials):
                    eeg = t["eeg"].unsqueeze(0).to(device)       
                    audio_a = t["audio_a"].unsqueeze(0).to(device) 
                    audio_b = t["audio_b"].unsqueeze(0).to(device)
                    
                    audio_a = audio_a.mean(dim=1, keepdim=True)
                    audio_b = audio_b.mean(dim=1, keepdim=True)
                    
                    eeg_mean = eeg.mean(dim=2, keepdim=True)
                    eeg_std = eeg.std(dim=2, keepdim=True) + 1e-8
                    eeg_norm = (eeg - eeg_mean) / eeg_std
                    
                    audio_a_mean = audio_a.mean(dim=2, keepdim=True)
                    audio_a_std = audio_a.std(dim=2, keepdim=True) + 1e-8
                    audio_a_norm = (audio_a - audio_a_mean) / audio_a_std
                    
                    audio_b_mean = audio_b.mean(dim=2, keepdim=True)
                    audio_b_std = audio_b.std(dim=2, keepdim=True) + 1e-8
                    audio_b_norm = (audio_b - audio_b_mean) / audio_b_std
                    
                    pred = model(eeg_norm)
                    
                    target = audio_a_norm.squeeze(1)
                    loss_test = custom_loss(pred, target, mse_weight=0.5, corr_weight=0.5).item()
                    test_loss += loss_test
                    total_test_samples += 1
                    
                    pred_np = pred.squeeze(0).cpu().numpy()
                    wav_a_np = audio_a_norm.squeeze(1).squeeze(0).cpu().numpy()
                    wav_b_np = audio_b_norm.squeeze(1).squeeze(0).cpu().numpy()
                    
                    c_att = safe_corr_np(pred_np, wav_a_np)
                    c_unatt = safe_corr_np(pred_np, wav_b_np)
                    test_corr_att += c_att
                    test_corr_unatt += c_unatt
                    trial_margin = c_att - c_unatt
                    trial_margins.append(trial_margin)
                    
                    trial_ok, n_win, c_win = evaluate_trial_majority_vote(pred_np, wav_a_np, wav_b_np, window_seconds=10, hop_seconds=1.0, fs=64)
                    if trial_ok: mv_trial_correct += 1
                    mv_windows_total += n_win
                    mv_windows_correct += c_win
                    
            test_loss /= total_test_samples
            test_corr_att /= total_test_samples
            test_corr_unatt /= total_test_samples
            
            trial_acc = mv_trial_correct / total_test_samples if total_test_samples > 0 else 0
            win_acc = mv_windows_correct / mv_windows_total if mv_windows_total > 0 else 0
            
            test_metrics = {
                "trial_accuracy": float(trial_acc),
                "window_accuracy": float(win_acc),
                "mean_pearson_att": float(test_corr_att),
                "mean_pearson_unatt": float(test_corr_unatt),
                "mean_margin": float(np.mean(trial_margins)),
                "median_margin": float(np.median(trial_margins)),
                "margin_std": float(np.std(trial_margins)),
                "positive_margin_fraction": float(np.sum(np.array(trial_margins) > 0) / len(trial_margins)),
                "negative_margin_fraction": float(np.sum(np.array(trial_margins) < 0) / len(trial_margins)),
                "fold_trial_margins": [float(m) for m in trial_margins],
                "best_epoch": best_epoch,
                "stopped_epoch": stopped_epoch,
                "history": fold_history
            }
                    
            loso_results[test_subject] = test_metrics
            print(f"--- Fold {test_subject} Completed ---")
            print(f"Test Trial Acc: {test_metrics['trial_accuracy']*100:.1f}%")
            print(f"Test Median Margin: {test_metrics['median_margin']:.4f}")
            
        all_seeds_results[str(seed)] = loso_results
        
    print("\n====================================================")
    print("ALL FOLDS COMPLETED FOR ALL SEEDS")
    print("====================================================")
    
    summary_file = out_dir / "conformer_loso_multiseed_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(all_seeds_results, f, indent=4)
        
    print(f"Saved Multi-Seed LOSO summary to {summary_file}")

if __name__ == "__main__":
    main()
