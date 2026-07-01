import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import json
import random

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

def custom_multitask_loss(pred, Ya_batch, Yb_batch, conf_pred, is_corrupted=None, lambda_conf=0.1, mse_weight=0.5, corr_weight=0.5):
    # Regression Loss (Primary)
    mse = nn.functional.mse_loss(pred, Ya_batch)
    corr_a = safe_corr_torch(pred, Ya_batch)
    reg_corr_loss = 1.0 - corr_a.mean()
    
    # If the batch is mixed with corrupted samples, we don't want the regression loss to be penalized by them
    # But for simplicity, we just compute it over the whole batch. The model's regression weights are largely frozen anyway.
    reg_loss = mse_weight * mse + corr_weight * reg_corr_loss
    
    # Confidence Target Generation
    corr_b = safe_corr_torch(pred, Yb_batch)
    margin = corr_a - corr_b
    
    # Target is 1 if margin > 0 (correct), else 0
    correct = (margin > 0).float()
    
    # OUTLIER EXPOSURE: If a sample is corrupted, its confidence target is strictly forced to 0
    if is_corrupted is not None:
        correct = correct * (1.0 - is_corrupted.float())
    
    # Confidence Loss (Auxiliary)
    conf_pred = conf_pred.squeeze()
    conf_pred = torch.clamp(conf_pred, 1e-7, 1.0 - 1e-7)
    conf_loss = nn.functional.binary_cross_entropy(conf_pred, correct)
    
    total_loss = reg_loss + lambda_conf * conf_loss
    
    return total_loss, reg_loss, conf_loss

def prepare_multitask_data(subject_data, window_sec=2, hop_sec=1, fs=64):
    win_samples = int(window_sec * fs)
    hop_samples = int(hop_sec * fs)
    
    X, Y_a, Y_b = [], [], []
    
    for t in subject_data:
        eeg = t["eeg"]       # (8, Time)
        audio_a = t["audio_a"] # (28, Time)
        audio_b = t["audio_b"] # (28, Time)
        
        # Audio preprocessing: mean over the 28 subbands to get a single broad envelope
        audio_a = audio_a.mean(dim=0, keepdim=True) # (1, Time)
        audio_b = audio_b.mean(dim=0, keepdim=True) # (1, Time)
        
        n_windows = (eeg.shape[1] - win_samples) // hop_samples + 1
        for i in range(max(1, n_windows)):
            start = i * hop_samples
            stop = start + win_samples
            if stop > eeg.shape[1]:
                break
                
            e = eeg[:, start:stop]
            a = audio_a[:, start:stop]
            b = audio_b[:, start:stop]
            
            a_mean = a.mean(dim=1, keepdim=True)
            a_std = a.std(dim=1, keepdim=True) + 1e-8
            a_norm = (a - a_mean) / a_std
            
            b_mean = b.mean(dim=1, keepdim=True)
            b_std = b.std(dim=1, keepdim=True) + 1e-8
            b_norm = (b - b_mean) / b_std
            
            e_mean = e.mean(dim=1, keepdim=True)
            e_std = e.std(dim=1, keepdim=True) + 1e-8
            e_norm = (e - e_mean) / e_std
            
            X.append(e_norm)
            Y_a.append(a_norm.squeeze(0)) # Squeeze to [Time]
            Y_b.append(b_norm.squeeze(0))
            
    if not X:
        return None
        
    return torch.stack(X), torch.stack(Y_a), torch.stack(Y_b)

def evaluate_fold(model, val_loader, device):
    model.eval()
    val_loss = 0.0
    val_reg_loss = 0.0
    val_conf_loss = 0.0
    margins = []
    
    with torch.no_grad():
        for eeg, ya, yb in val_loader:
            eeg, ya, yb = eeg.to(device), ya.to(device), yb.to(device)
            pred, z_pool = model(eeg, return_features=True)
            
            corr_a = safe_corr_torch(pred, ya)
            corr_b = safe_corr_torch(pred, yb)
            margin = corr_a - corr_b
            
            conf_pred = model.predict_confidence(z_pool, corr_a, corr_b, margin)
            
            loss, reg, conf = custom_multitask_loss(pred, ya, yb, conf_pred)
            val_loss += loss.item()
            val_reg_loss += reg.item()
            val_conf_loss += conf.item()
            
            margins.extend(margin.cpu().numpy())
            
    val_loss /= len(val_loader)
    val_reg_loss /= len(val_loader)
    val_conf_loss /= len(val_loader)
    mean_margin = np.mean(margins)
    
    return val_loss, val_reg_loss, val_conf_loss, mean_margin

def main():
    print("--- Phase 7: Multi-Task Confidence Training (LOSO) ---")
    
    cache_dir = REPO_ROOT / "data" / "processed_kul"
    if Path("/kaggle/input/datasets/lowk1ee/kul-preprocessed-cache/data/processed_kul").exists():
        cache_dir = Path("/kaggle/input/datasets/lowk1ee/kul-preprocessed-cache/data/processed_kul")
    elif Path("/kaggle/input/datasets/lokeshgile/kul-processed/data/processed_kul").exists():
        cache_dir = Path("/kaggle/input/datasets/lokeshgile/kul-processed/data/processed_kul")
        
    try:
        loader = KULCachedLoader(cache_dir)
        all_subject_data = loader.load_all()
    except FileNotFoundError:
        print(f"KUL Cache not found at {cache_dir}.")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    out_dir = REPO_ROOT / "results" / "run7_multitask_conformer_loso"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    subjects = sorted(list(all_subject_data.keys()))
    
    # We will use only seed 1 for the Phase 7 benchmark to save compute time
    seed = 1
    print(f"STARTING MULTI-TASK SEED: {seed}")
    
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
        print(f"Starting Multi-Task LOSO Fold: Test Subject = {test_subject}")
        print(f"{'='*60}")
        
        remaining_subjects = [s for s in subjects if s != test_subject]
        test_idx = subjects.index(test_subject)
        val_subject = remaining_subjects[test_idx % len(remaining_subjects)]
        
        train_trials = []
        for sub in remaining_subjects:
            if sub != val_subject:
                train_trials.extend(all_subject_data[sub])
        val_trials = all_subject_data[val_subject]
                
        train_tensors = prepare_multitask_data(train_trials, window_sec=2, hop_sec=1, fs=64)
        val_tensors = prepare_multitask_data(val_trials, window_sec=2, hop_sec=2, fs=64)
        
        if train_tensors is None or val_tensors is None:
            print("Missing data tensors. Skipping.")
            continue
            
        X_train, Ya_train, Yb_train = train_tensors
        X_val, Ya_val, Yb_val = val_tensors
        
        train_loader = DataLoader(TensorDataset(X_train, Ya_train, Yb_train), batch_size=128, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, Ya_val, Yb_val), batch_size=128, shuffle=False)
        
        model = AADConformer(in_channels=8).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        epochs = 15
        best_val_margin = -float('inf')
        ckpt_path = checkpoint_dir / f"model_{test_subject}.pt"
        
        # Load pre-trained regression weights to speed up training
        # This acts as fine-tuning and stabilizes the latent space for the confidence head
        pretrain_ckpt = REPO_ROOT / "conformer_loso_results" / "checkpoints" / "seed_1" / f"model_{test_subject}.pt"
        kaggle_ckpt = Path("/kaggle/input/datasets/lowkieee/eeg-aad-conformer-seed1-checkpoints/checkpoints/seed_1") / f"model_{test_subject}.pt"
        
        loaded_pretrain = False
        if kaggle_ckpt.exists():
            model.load_state_dict(torch.load(kaggle_ckpt, map_location=device), strict=False)
            loaded_pretrain = True
        elif pretrain_ckpt.exists():
            model.load_state_dict(torch.load(pretrain_ckpt, map_location=device), strict=False)
            loaded_pretrain = True
            
        if loaded_pretrain:
            print(f"Successfully loaded pre-trained Phase 4 regression weights! Fine-tuning for 5 epochs...")
            epochs = 5
            optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4) # Lower LR for fine-tuning
        
        for ep in range(epochs):
            model.train()
            train_loss = 0.0
            train_reg = 0.0
            train_conf = 0.0
            
            for eeg, ya, yb in train_loader:
                eeg, ya, yb = eeg.to(device), ya.to(device), yb.to(device)
                optimizer.zero_grad()
                
                # OUTLIER EXPOSURE DATA AUGMENTATION
                # We corrupt 25% of the batch to teach the confidence head to reject bad EEG
                B = eeg.size(0)
                is_corrupted = torch.zeros(B, device=device, dtype=torch.bool)
                num_corrupt = int(B * 0.25)
                
                if num_corrupt > 0:
                    corrupt_idx = torch.randperm(B)[:num_corrupt]
                    is_corrupted[corrupt_idx] = True
                    
                    # 50% Random Noise, 50% Zeros
                    half = num_corrupt // 2
                    eeg[corrupt_idx[:half]] = torch.randn_like(eeg[corrupt_idx[:half]])
                    eeg[corrupt_idx[half:]] = torch.zeros_like(eeg[corrupt_idx[half:]])
                
                pred, z_pool = model(eeg, return_features=True)
                
                corr_a = safe_corr_torch(pred, ya)
                corr_b = safe_corr_torch(pred, yb)
                margin = corr_a - corr_b
                
                conf_pred = model.predict_confidence(z_pool, corr_a, corr_b, margin)
                
                loss, reg, conf = custom_multitask_loss(pred, ya, yb, conf_pred, is_corrupted=is_corrupted)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_reg += reg.item()
                train_conf += conf.item()
                
            train_loss /= len(train_loader)
            train_reg /= len(train_loader)
            train_conf /= len(train_loader)
            
            val_loss, val_reg, val_conf, val_margin = evaluate_fold(model, val_loader, device)
            
            print(f"Epoch {ep+1:02d} | Train Loss: {train_loss:.4f} (Reg:{train_reg:.4f} Conf:{train_conf:.4f}) "
                  f"| Val Loss: {val_loss:.4f} (Reg:{val_reg:.4f} Conf:{val_conf:.4f}) | Val Margin: {val_margin:.4f}")
                  
            if val_margin > best_val_margin:
                best_val_margin = val_margin
                torch.save(model.state_dict(), ckpt_path)
                
        print(f"Fold completed. Best Val Margin: {best_val_margin:.4f}")
        loso_results[test_subject] = float(best_val_margin)
        
    print("\n--- Phase 7 Multi-Task Training Summary ---")
    mean_margin = np.mean(list(loso_results.values()))
    print(f"Mean Validation Margin across 16 subjects: {mean_margin:.4f}")
    
    with open(out_dir / f"multitask_conformer_val_results.json", "w") as f:
        json.dump(loso_results, f, indent=4)
        
if __name__ == "__main__":
    main()
