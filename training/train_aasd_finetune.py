import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.aad_conformer import AADConformer

def chunk_aasd_data(eeg, audio_l, audio_r, raw_evs, window_sec=2.0, hop_sec=1.0, fs=64):
    """
    Chunks a trial into windows. Uses raw_evs to determine the ground truth attention
    for each window. Discards windows where attention switches.
    """
    win_samples = int(window_sec * fs)
    hop_samples = int(hop_sec * fs)
    
    # Generate GT at 64Hz
    t_array = np.arange(eeg.shape[1]) / float(fs) + 1.0 # align with previous scripts
    gt = np.zeros(len(t_array))
    
    st_times = []
    types = []
    for ev_t, ev_lat in raw_evs:
        if ev_t in ['179', '184', '254', '255']:
            st_times.append(ev_lat / 128.0)
            types.append('R' if ev_t in ['179', '254'] else 'L') # Target speaker B
            
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
        
        # Only use clean windows
        mean_gt = np.mean(gt_chunk)
        if mean_gt == 1.0: # Right
            att = audio_r[:, start:end]
            unatt = audio_l[:, start:end]
            chunks_eeg.append(eeg[:, start:end])
            chunks_att.append(att)
            chunks_unatt.append(unatt)
        elif mean_gt == 0.0: # Left
            att = audio_l[:, start:end]
            unatt = audio_r[:, start:end]
            chunks_eeg.append(eeg[:, start:end])
            chunks_att.append(att)
            chunks_unatt.append(unatt)
            
        start += hop_samples
        
    return chunks_eeg, chunks_att, chunks_unatt

def pearson_loss(pred_att, true_att):
    cos_sim = torch.sum(pred_att * true_att, dim=-1) / (torch.norm(pred_att, dim=-1) * torch.norm(true_att, dim=-1) + 1e-8)
    return -torch.mean(cos_sim)

def main():
    print("[INFO] Starting Phase 28 Domain Adaptation: AASD Fine-Tuning")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache_dir = REPO_ROOT / 'data' / 'processed_aasd'
    
    import glob
    cache_files = glob.glob(str(cache_dir / "*.pt"))
    subjects = sorted([os.path.splitext(os.path.basename(f))[0] for f in cache_files])
    
    test_subs = ['S18']
    train_subs = [s for s in subjects if s not in test_subs]
    
    print(f"[INFO] Training on {len(train_subs)} subjects, Testing on {len(test_subs)} subjects")
    
    X_train, y_att_train, y_unatt_train = [], [], []
    X_test, y_att_test, y_unatt_test = [], [], []
    
    for sub in subjects:
        cache_file = cache_dir / f"{sub}.pt"
        if not cache_file.exists():
            print(f"[ERROR] Cache not found for {sub}")
            continue
            
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
            
        print(f"[{sub}] Extracted {len(sub_X)} clean windows.")
        
        if sub in train_subs:
            X_train.extend(sub_X)
            y_att_train.extend(sub_ya)
            y_unatt_train.extend(sub_yb)
        else:
            X_test.extend(sub_X)
            y_att_test.extend(sub_ya)
            y_unatt_test.extend(sub_yb)
            
    X_train = torch.FloatTensor(np.array(X_train))
    y_att_train = torch.FloatTensor(np.array(y_att_train))
    y_unatt_train = torch.FloatTensor(np.array(y_unatt_train))
    
    X_test = torch.FloatTensor(np.array(X_test))
    y_att_test = torch.FloatTensor(np.array(y_att_test))
    y_unatt_test = torch.FloatTensor(np.array(y_unatt_test))
    
    train_ds = TensorDataset(X_train, y_att_train, y_unatt_train)
    test_ds = TensorDataset(X_test, y_att_test, y_unatt_test)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    model = AADConformer(in_channels=8).to(device)
    
    # Load KUL pre-trained weights
    ckpt_path = '/kaggle/input/datasets/lowkieee/eeg-aad-conformer-seed1-checkpoints/checkpoints/seed_1/model_S1.pt'
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
        print(f"[INFO] Loaded KUL pre-trained weights from {ckpt_path}")
    else:
        print("[WARNING] KUL weights not found! Training from scratch.")
        
    # Freeze audio encoder
    # Assuming AADConformer has no explicit separate audio encoder? 
    # Wait, in the evaluation script we just use: out, _ = model(eeg_t, return_features=True)
    # And correlate `out` with audio. AADConformer predicts the envelope directly.
    # There is no Audio Encoder to freeze! The model just maps EEG -> Envelope.
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # Higher LR to adapt
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    epochs = 30
    best_test_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 5
    
    print("\nStarting Training...")
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
            
            # Remove channel dimension
            out = out.squeeze(1)
            att = att.squeeze(1)
            unatt = unatt.squeeze(1)
            
            loss = pearson_loss(out, att)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * eeg.size(0)
            
            # Train Accuracy
            with torch.no_grad():
                dist_att = 1.0 - torch.sum(out * att, dim=-1) / (torch.norm(out, dim=-1) * torch.norm(att, dim=-1) + 1e-8)
                dist_unatt = 1.0 - torch.sum(out * unatt, dim=-1) / (torch.norm(out, dim=-1) * torch.norm(unatt, dim=-1) + 1e-8)
                train_correct += (dist_att < dist_unatt).sum().item()
                train_total += eeg.size(0)
            
        train_loss /= len(train_ds)
        train_acc = train_correct / train_total
        
        # Eval
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for eeg, att, unatt in test_loader:
                eeg = eeg.to(device)
                att = att.to(device)
                unatt = unatt.to(device)
                
                out, _ = model(eeg, return_features=True)
                out = out.squeeze(1)
                att = att.squeeze(1)
                unatt = unatt.squeeze(1)
                
                loss = pearson_loss(out, att)
                test_loss += loss.item() * eeg.size(0)
                
                # Accuracy
                dist_att = 1.0 - torch.sum(out * att, dim=-1) / (torch.norm(out, dim=-1) * torch.norm(att, dim=-1) + 1e-8)
                dist_unatt = 1.0 - torch.sum(out * unatt, dim=-1) / (torch.norm(out, dim=-1) * torch.norm(unatt, dim=-1) + 1e-8)
                correct += (dist_att < dist_unatt).sum().item()
                total += eeg.size(0)
                
        test_loss /= len(test_ds)
        test_acc = correct / total
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - Test Loss: {test_loss:.4f} - Test Acc: {test_acc:.4f}")
        
        scheduler.step(test_loss)
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            out_dir = REPO_ROOT / 'checkpoints' / 'aasd_finetuned'
            out_dir.mkdir(parents=True, exist_ok=True)
            save_path = out_dir / 'model_S18_loso.pt'
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved new best model to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"[INFO] Early stopping triggered after {epoch+1} epochs.")
                break
                
    print(f"\n[INFO] Finished Fine-Tuning. Best Test Loss: {best_test_loss:.4f}")

if __name__ == "__main__":
    main()
