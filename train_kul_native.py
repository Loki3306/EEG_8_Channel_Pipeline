import os
import sys
import math
import numpy as np
import scipy.io
import scipy.signal
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.matchnet import ContrastiveMatchNet, contrastive_loss

def get_kul_trials():
    mat_path = "/kaggle/input/datasets/lowk1ee/s1-klu/S1_KLU.mat"
    if not os.path.exists(mat_path):
        mat_path = "data/S1_KLU.mat"
    if not os.path.exists(mat_path):
        raise FileNotFoundError("Missing KUL data.")
        
    mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    trials = mat['trials'] if 'trials' in mat else mat['trial']
    return trials

def preprocess_trial(trial, envelope_cache, apply_car=True):
    eeg_data = trial.RawData.EegData
    fs_eeg = trial.FileHeader.SampleRate
    channel_names = [ch.Label for ch in trial.FileHeader.Channels]
    
    target_channels = ['T7', 'C2', 'FT8', 'P7', 'CPz', 'Fp1', 'TP8', 'C3']
    
    if apply_car:
        eeg_data = eeg_data - eeg_data.mean(axis=1, keepdims=True)
        
    try:
        sel_idx = [channel_names.index(tc) if tc in channel_names else [c.upper() for c in channel_names].index(tc.upper()) for tc in target_channels]
    except ValueError:
        return None, None, None
        
    eeg_8 = eeg_data[:, sel_idx]
    
    nyq = 0.5 * fs_eeg
    b, a = scipy.signal.butter(4, [1.0/nyq, 8.0/nyq], btype='band')
    eeg_8 = scipy.signal.filtfilt(b, a, eeg_8, axis=0)
    
    g = math.gcd(64, int(fs_eeg))
    eeg_8 = scipy.signal.resample_poly(eeg_8, 64 // g, int(fs_eeg) // g, axis=0)
    
    arr = eeg_8 - eeg_8.mean(axis=0, keepdims=True)
    scale = arr.std(axis=0, keepdims=True) + 1e-12
    eeg_norm = arr / scale
    
    att_ear = trial.attended_ear
    stimuli = trial.stimuli
    if len(stimuli) < 2: return None, None, None
    att_wav_name = str(stimuli[0] if att_ear == 'L' else stimuli[1]).strip()
    unatt_wav_name = str(stimuli[1] if att_ear == 'L' else stimuli[0]).strip()
    
    def find_wav(name):
        wav_dir = "/kaggle/input/datasets/lowk1ee/audio-klu" if os.path.exists("/kaggle/input/datasets/lowk1ee/audio-klu") else "data"
        for r, d, f in os.walk(wav_dir):
            for file in f:
                if name in file and file.endswith(".wav"):
                    return os.path.join(r, file)
        return None
        
    att_wav_path = find_wav(att_wav_name)
    unatt_wav_path = find_wav(unatt_wav_name)
    
    if att_wav_path and unatt_wav_path and att_wav_path in envelope_cache and unatt_wav_path in envelope_cache:
        env_att = envelope_cache[att_wav_path]
        env_unatt = envelope_cache[unatt_wav_path]
        
        def norm_env(env):
            env = env.T
            env = env - env.mean(axis=0, keepdims=True)
            env = env / (env.std(axis=0, keepdims=True) + 1e-12)
            return env.T
            
        env_att = norm_env(env_att)
        env_unatt = norm_env(env_unatt)
        
        min_len = min(len(eeg_norm), env_att.shape[1], env_unatt.shape[1])
        return eeg_norm[:min_len].T, env_att[:, :min_len], env_unatt[:, :min_len]
        
    return None, None, None

def chunk_data(x, ya, yb, window_sec, hop_sec, fs=64):
    win_samples = int(window_sec * fs)
    hop_samples = int(hop_sec * fs)
    chunks_x, chunks_ya, chunks_yb = [], [], []
    start = 0
    while start + win_samples <= x.shape[1]:
        end = start + win_samples
        chunks_x.append(x[:, start:end])
        chunks_ya.append(ya[:, start:end])
        chunks_yb.append(yb[:, start:end])
        start += hop_samples
    return chunks_x, chunks_ya, chunks_yb

def evaluate_val(model, val_trials_data, device, window_sec=10, fs=64):
    model.eval()
    win_samples = int(window_sec * fs)
    n_correct = 0.0
    n_total = 0
    
    with torch.no_grad():
        for x, ya, yb in val_trials_data:
            start = 0
            while start + win_samples <= x.shape[1]:
                end = start + win_samples
                cx = torch.FloatTensor(x[:, start:end]).unsqueeze(0).to(device)
                cya = torch.FloatTensor(ya[:, start:end]).unsqueeze(0).to(device)
                cyb = torch.FloatTensor(yb[:, start:end]).unsqueeze(0).to(device)
                
                z_eeg, z_a, z_b = model(cx, cya, cyb)
                sim_a = F.cosine_similarity(z_eeg, z_a, dim=1).mean().item()
                sim_b = F.cosine_similarity(z_eeg, z_b, dim=1).mean().item()
                
                if sim_a > sim_b: n_correct += 1.0
                elif sim_a == sim_b: n_correct += 0.5
                n_total += 1
                start += win_samples
                
    return n_correct / max(n_total, 1)

def train_kul_native():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training KUL-Native MatchNet on {device}...")
    
    cache_path = "kul_gammatone_cache.pkl"
    if os.path.exists(cache_path):
        import pickle
        with open(cache_path, "rb") as f:
            envelope_cache = pickle.load(f)
    else:
        print("Missing kul_gammatone_cache.pkl. Run experiment 18 first to generate it.")
        return
        
    trials = get_kul_trials()
    
    print(f"Loaded {len(trials)} trials from KUL S1. Splitting 15 Train / 5 Val.")
    # For a robust split, we take trials 0-14 for training, 15-19 for val.
    train_trials = trials[:15]
    val_trials = trials[15:]
    
    tr_x_chunks, tr_ya_chunks, tr_yb_chunks = [], [], []
    val_data = []
    
    for i, t in enumerate(train_trials):
        x, ya, yb = preprocess_trial(t, envelope_cache, apply_car=True)
        if x is not None:
            cx, cya, cyb = chunk_data(x, ya, yb, window_sec=5, hop_sec=2)
            tr_x_chunks.extend(cx)
            tr_ya_chunks.extend(cya)
            tr_yb_chunks.extend(cyb)
            
    for i, t in enumerate(val_trials):
        x, ya, yb = preprocess_trial(t, envelope_cache, apply_car=True)
        if x is not None:
            val_data.append((x, ya, yb))
            
    print(f"Generated {len(tr_x_chunks)} training chunks (5s).")
    
    X_tr_t = torch.FloatTensor(np.stack(tr_x_chunks))
    YA_tr_t = torch.FloatTensor(np.stack(tr_ya_chunks))
    YB_tr_t = torch.FloatTensor(np.stack(tr_yb_chunks))
    
    train_dataset = TensorDataset(X_tr_t, YA_tr_t, YB_tr_t)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True)
    
    model = ContrastiveMatchNet(eeg_model_type="eegnet", eeg_channels=8, audio_channels=28, latent_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    
    best_val_acc = 0.0
    best_weights = deepcopy(model.state_dict())
    patience = 10
    epochs_no_improve = 0
    
    for epoch in range(100):
        model.train()
        train_loss = 0.0
        
        for bx, bya, byb in train_loader:
            bx = bx.to(device, non_blocking=True)
            bya = bya.to(device, non_blocking=True)
            byb = byb.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                z_eeg, z_a, z_b = model(bx, bya, byb)
                loss, sa, sb = contrastive_loss(z_eeg, z_a, z_b, margin=0.1)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
        avg_loss = train_loss / len(train_loader)
        val_acc = evaluate_val(model, val_data, device, window_sec=10)
        
        print(f"Epoch {epoch+1:02d}/100 | Train Loss: {avg_loss:.4f} | Val Acc (10s): {val_acc*100:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break
            
    os.makedirs("checkpoints", exist_ok=True)
    best_path = "checkpoints/matchnet_kul_native_best.pth"
    torch.save(best_weights, best_path)
    print(f"\nTraining Complete. Best KUL-Native Val Acc: {best_val_acc*100:.2f}%")
    print(f"Model saved to {best_path}")

if __name__ == "__main__":
    train_kul_native()
