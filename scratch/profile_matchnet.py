import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.matchnet import ContrastiveMatchNet, contrastive_loss

# Mock Data parameters
N_SAMPLES = 2000
SEQ_LEN = 320 # 5 seconds at 64Hz
EEG_CHANNELS = 8
AUDIO_CHANNELS = 28

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_mock_data():
    X = np.random.randn(N_SAMPLES, EEG_CHANNELS, SEQ_LEN).astype(np.float32)
    YA = np.random.randn(N_SAMPLES, AUDIO_CHANNELS, SEQ_LEN).astype(np.float32)
    YB = np.random.randn(N_SAMPLES, AUDIO_CHANNELS, SEQ_LEN).astype(np.float32)
    return X, YA, YB

def profile_current_baseline():
    print("--- 1. Current Baseline (Manual Batching) ---")
    X, YA, YB = get_mock_data()
    model = ContrastiveMatchNet(eeg_model_type='eegnet', eeg_channels=EEG_CHANNELS, audio_channels=AUDIO_CHANNELS, latent_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    batch_size = 16
    perm = np.arange(len(X))
    
    # Warmup
    for i in range(0, 32, batch_size):
        bx = torch.FloatTensor(np.stack([X[j] for j in perm[i:i+batch_size]])).to(device)
        bya = torch.FloatTensor(np.stack([YA[j] for j in perm[i:i+batch_size]])).to(device)
        byb = torch.FloatTensor(np.stack([YB[j] for j in perm[i:i+batch_size]])).to(device)
        z_eeg, z_a, z_b = model(bx, bya, byb)
        loss, _, _ = contrastive_loss(z_eeg, z_a, z_b, margin=0.1)
        loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    times_data, times_fwd, times_bwd, times_opt = [], [], [], []
    
    for i in range(0, 400, batch_size):
        t0 = time.time()
        bx = torch.FloatTensor(np.stack([X[j] for j in perm[i:i+batch_size]])).to(device)
        bya = torch.FloatTensor(np.stack([YA[j] for j in perm[i:i+batch_size]])).to(device)
        byb = torch.FloatTensor(np.stack([YB[j] for j in perm[i:i+batch_size]])).to(device)
        torch.cuda.synchronize()
        t1 = time.time()
        
        optimizer.zero_grad()
        z_eeg, z_a, z_b = model(bx, bya, byb)
        loss, _, _ = contrastive_loss(z_eeg, z_a, z_b, margin=0.1)
        torch.cuda.synchronize()
        t2 = time.time()
        
        loss.backward()
        torch.cuda.synchronize()
        t3 = time.time()
        
        optimizer.step()
        torch.cuda.synchronize()
        t4 = time.time()
        
        times_data.append(t1 - t0)
        times_fwd.append(t2 - t1)
        times_bwd.append(t3 - t2)
        times_opt.append(t4 - t3)
        
    print(f"Data Load: {np.mean(times_data)*1000:.2f} ms")
    print(f"Forward:   {np.mean(times_fwd)*1000:.2f} ms")
    print(f"Backward:  {np.mean(times_bwd)*1000:.2f} ms")
    print(f"Optimizer: {np.mean(times_opt)*1000:.2f} ms")
    
    total_time = sum(times_data) + sum(times_fwd) + sum(times_bwd) + sum(times_opt)
    samples_processed = len(times_data) * batch_size
    print(f"Baseline Throughput: {samples_processed / total_time:.2f} samples/sec\n")

def test_dataloader(num_workers, pin_memory):
    print(f"--- DataLoader: workers={num_workers}, pin_memory={pin_memory} ---")
    X, YA, YB = get_mock_data()
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(YA), torch.from_numpy(YB))
    loader = DataLoader(dataset, batch_size=128, shuffle=True, 
                        num_workers=num_workers, pin_memory=pin_memory,
                        persistent_workers=True if num_workers > 0 else False)
    
    model = ContrastiveMatchNet(eeg_model_type='eegnet', eeg_channels=EEG_CHANNELS, audio_channels=AUDIO_CHANNELS, latent_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Warmup
    for bx, bya, byb in loader:
        bx, bya, byb = bx.to(device), bya.to(device), byb.to(device)
        z_eeg, z_a, z_b = model(bx, bya, byb)
        loss, _, _ = contrastive_loss(z_eeg, z_a, z_b, margin=0.1)
        loss.backward()
        optimizer.step()
        break
        
    torch.cuda.synchronize()
    start = time.time()
    samples = 0
    for bx, bya, byb in loader:
        bx, bya, byb = bx.to(device, non_blocking=True), bya.to(device, non_blocking=True), byb.to(device, non_blocking=True)
        optimizer.zero_grad()
        z_eeg, z_a, z_b = model(bx, bya, byb)
        loss, _, _ = contrastive_loss(z_eeg, z_a, z_b, margin=0.1)
        loss.backward()
        optimizer.step()
        samples += len(bx)
        
    torch.cuda.synchronize()
    end = time.time()
    print(f"Throughput: {samples / (end - start):.2f} samples/sec\n")

def test_amp():
    print("--- Testing AMP (Mixed Precision) ---")
    X, YA, YB = get_mock_data()
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(YA), torch.from_numpy(YB))
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
    
    model = ContrastiveMatchNet(eeg_model_type='eegnet', eeg_channels=EEG_CHANNELS, audio_channels=AUDIO_CHANNELS, latent_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler()
    
    torch.cuda.synchronize()
    start = time.time()
    samples = 0
    for bx, bya, byb in loader:
        bx, bya, byb = bx.to(device, non_blocking=True), bya.to(device, non_blocking=True), byb.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            z_eeg, z_a, z_b = model(bx, bya, byb)
            loss, _, _ = contrastive_loss(z_eeg, z_a, z_b, margin=0.1)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        samples += len(bx)
        
    torch.cuda.synchronize()
    end = time.time()
    print(f"AMP Throughput: {samples / (end - start):.2f} samples/sec\n")
    
def test_batch_size(batch_size):
    print(f"--- Testing Batch Size: {batch_size} ---")
    X, YA, YB = get_mock_data()
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(YA), torch.from_numpy(YB))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
    
    model = ContrastiveMatchNet(eeg_model_type='eegnet', eeg_channels=EEG_CHANNELS, audio_channels=AUDIO_CHANNELS, latent_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler()
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.time()
    samples = 0
    for bx, bya, byb in loader:
        bx, bya, byb = bx.to(device, non_blocking=True), bya.to(device, non_blocking=True), byb.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            z_eeg, z_a, z_b = model(bx, bya, byb)
            loss, _, _ = contrastive_loss(z_eeg, z_a, z_b, margin=0.1)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        samples += len(bx)
        
    torch.cuda.synchronize()
    end = time.time()
    mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    print(f"Throughput: {samples / (end - start):.2f} samples/sec | Peak Mem: {mem:.2f} MB\n")

if __name__ == "__main__":
    profile_current_baseline()
    torch.backends.cudnn.benchmark = True
    print("Enabled CuDNN Benchmark\n")
    test_dataloader(0, False)
    test_dataloader(2, True)
    test_dataloader(4, True)
    test_amp()
    test_batch_size(16)
    test_batch_size(64)
    test_batch_size(128)
    test_batch_size(256)
    test_batch_size(512)
