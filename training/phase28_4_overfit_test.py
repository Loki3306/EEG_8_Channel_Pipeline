import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.aad_conformer import AADConformer

def chunk_aasd_data(eeg, audio_l, audio_r, raw_evs, window_sec=2.0, hop_sec=1.0, fs=64):
    win_samples = int(window_sec * fs)
    hop_samples = int(hop_sec * fs)
    t_array = np.arange(eeg.shape[1]) / float(fs) + 1.0 
    gt = np.zeros(len(t_array))
    
    st_times = []
    types = []
    for ev_t, ev_lat in raw_evs:
        if ev_t in ['179', '184', '254', '255']:
            st_times.append(ev_lat / 128.0)
            types.append('R' if ev_t in ['179', '254'] else 'L')
            
    if len(types) > 0:
        current_state = 1 if types[0] == 'R' else 0
        for i, t in enumerate(t_array):
            state = current_state
            for st, s_type in zip(st_times, types):
                if t >= st:
                    state = 1 if s_type == 'R' else 0
            gt[i] = state
            
    chunks_eeg, chunks_att, chunks_unatt = [], [], []
    start = 0
    while start + win_samples <= eeg.shape[1]:
        end = start + win_samples
        gt_chunk = gt[start:end]
        mean_gt = np.mean(gt_chunk)
        if mean_gt == 1.0:
            chunks_eeg.append(eeg[:, start:end])
            chunks_att.append(audio_r[:, start:end])
            chunks_unatt.append(audio_l[:, start:end])
        elif mean_gt == 0.0:
            chunks_eeg.append(eeg[:, start:end])
            chunks_att.append(audio_l[:, start:end])
            chunks_unatt.append(audio_r[:, start:end])
        start += hop_samples
        
    return chunks_eeg, chunks_att, chunks_unatt

def pearson_loss(pred_att, true_att):
    cos_sim = torch.sum(pred_att * true_att, dim=-1) / (torch.norm(pred_att, dim=-1) * torch.norm(true_att, dim=-1) + 1e-8)
    return -torch.mean(cos_sim)

def main():
    print("[INFO] Starting Phase 28.4: Overfit Sanity Test")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache_dir = REPO_ROOT / 'data' / 'processed_aasd'
    cache_file = cache_dir / "S1.pt"
    
    if not cache_file.exists():
        print(f"[ERROR] Cache not found for S1. Run build_aasd_cache.py first.")
        return
        
    data = torch.load(cache_file, weights_only=False)
    trials = data['trials']
    
    sub_X, sub_ya, sub_yb = [], [], []
    for trial in trials:
        eeg = trial['eeg'].numpy()
        audio_l = trial['audio_l'].numpy()
        audio_r = trial['audio_r'].numpy()
        raw_evs = trial['meta']['raw_evs']
        
        c_e, c_a, c_u = chunk_aasd_data(eeg, audio_l, audio_r, raw_evs, window_sec=2.0, hop_sec=1.0)
        sub_X.extend(c_e)
        sub_ya.extend(c_a)
        sub_yb.extend(c_u)
        
    print(f"[INFO] Extracted {len(sub_X)} clean windows from S1.")
    
    # ISOLATE: Take only the first 100 windows
    NUM_WINDOWS = 100
    if len(sub_X) < NUM_WINDOWS:
        print("[WARNING] Less than 100 windows available. Using all available.")
        NUM_WINDOWS = len(sub_X)
        
    sub_X = sub_X[:NUM_WINDOWS]
    sub_ya = sub_ya[:NUM_WINDOWS]
    sub_yb = sub_yb[:NUM_WINDOWS]
    
    print(f"[INFO] Overfitting on {NUM_WINDOWS} windows.")
    
    X_train = torch.FloatTensor(np.array(sub_X))
    y_att_train = torch.FloatTensor(np.array(sub_ya))
    y_unatt_train = torch.FloatTensor(np.array(sub_yb))
    
    # We will use a batch size of 100 (full batch) to maximize gradient stability on a small dataset
    train_ds = TensorDataset(X_train, y_att_train, y_unatt_train)
    train_loader = DataLoader(train_ds, batch_size=NUM_WINDOWS, shuffle=True)
    
    # Initialize from scratch to rule out pre-trained weight issues
    model = AADConformer(in_channels=8).to(device)
    
    # High learning rate for fast overfitting
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 200
    
    print("\nStarting Training (Overfit Loop)...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for eeg, att, unatt in train_loader:
            eeg = eeg.to(device)
            att = att.to(device)
            unatt = unatt.to(device)
            
            optimizer.zero_grad()
            out, _ = model(eeg, return_features=True)
            
            out = out.squeeze(1)
            att = att.squeeze(1)
            unatt = unatt.squeeze(1)
            
            loss = pearson_loss(out, att)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * eeg.size(0)
            
            with torch.no_grad():
                dist_att = 1.0 - torch.sum(out * att, dim=-1) / (torch.norm(out, dim=-1) * torch.norm(att, dim=-1) + 1e-8)
                dist_unatt = 1.0 - torch.sum(out * unatt, dim=-1) / (torch.norm(out, dim=-1) * torch.norm(unatt, dim=-1) + 1e-8)
                train_correct += (dist_att < dist_unatt).sum().item()
                train_total += eeg.size(0)
                
        train_loss /= len(train_ds)
        train_acc = train_correct / train_total
        
        # Print every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}/{epochs} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f}")
            
    print("\n[INFO] Sanity Test Complete.")
    if train_acc >= 0.95:
        print("RESULT: SUCCESS. The model architecture can fit this dataset perfectly.")
    elif train_acc > 0.70:
        print("RESULT: PARTIAL. The model struggles slightly but can learn. Investigate data scale/labels.")
    else:
        print("RESULT: FAILURE. The model cannot even memorize 100 windows. Pipeline or architecture is mathematically broken.")

if __name__ == "__main__":
    main()
