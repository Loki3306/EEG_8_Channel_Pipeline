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
    # Both pred and target are [Batch, Time]
    mse = nn.functional.mse_loss(pred, target)
    corr = safe_corr_torch(pred, target)
    mean_corr = corr.mean()
    corr_loss = 1.0 - mean_corr
    return mse_weight * mse + corr_weight * corr_loss

def prepare_data(subject_data, window_sec=10, hop_sec=2, fs=64):
    win_samples = int(window_sec * fs)
    hop_samples = int(hop_sec * fs)
    
    X, Y_a = [], []
    
    for t in subject_data:
        eeg = t["eeg"]       # (8, Time)
        audio_a = t["audio_a"] # (28, Time) -> we will use the envelope of all bands by averaging
        
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
        subjects = sorted(list(all_subject_data.keys()))
        test_subject = subjects[0]
        print(f"Smoke Test using Train Subject = {test_subject}")
        train_trials = all_subject_data[test_subject]
        train_tensors = prepare_data(train_trials, window_sec=2, hop_sec=1, fs=64)
        if train_tensors is None:
            print(f"No training data for {test_subject}.")
            return
        X_train, Ya_train = train_tensors
    except FileNotFoundError:
        print(f"KUL Cache not found at {cache_dir}. USING MOCK DATA for smoke test.")
        # Mock data: Batch of 16, 8 channels, 2 seconds at 64Hz = 128 time steps
        X_train = torch.randn(16, 8, 128)
        Ya_train = torch.randn(16, 128)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataset = TensorDataset(X_train, Ya_train)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Initialize Conformer
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
    
    # Calculate parameter count
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {num_params}")
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    epochs = 2
    
    print("\nStarting Training...")
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
        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f}")
        
    print("\nSmoke Test Passed! Forward, backward, and shape checks are successful.")

if __name__ == "__main__":
    main()
