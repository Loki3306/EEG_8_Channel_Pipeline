import os
import sys
import numpy as np
import scipy.io
import scipy.signal
import torch
import glob
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.aad_conformer import AADConformer

def cross_corr(x, y):
    if len(x) == 0 or len(y) == 0: return 0.0
    x = (x - np.mean(x)) / (np.std(x) + 1e-8)
    y = (y - np.mean(y)) / (np.std(y) + 1e-8)
    c = np.corrcoef(x, y)[0, 1]
    return c if not np.isnan(c) else 0.0

def main():
    print("==================================================")
    print("=== PHASE 28.19 FORENSIC GROUND-TRUTH VERIFIER ===")
    print("==================================================\n")
    
    S1_mat = '/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG/S01/S1.mat'
    if not os.path.exists(S1_mat):
        matches = glob.glob('/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG/*/*.mat')
        if not matches:
            print("ERROR: S1.mat not found. Please run on Kaggle.")
            return
        S1_mat = matches[0]
        
    print(f"[MAT FILE] Loading {S1_mat}")
    mat = scipy.io.loadmat(S1_mat, squeeze_me=True, struct_as_record=False)
    eeg_var = [k for k in mat.keys() if not k.startswith('__')][0]
    
    data_all = mat[eeg_var].data
    events = mat[eeg_var].event
    
    print(f"├── EEG shape: {data_all.shape}")
    print(f"├── Total events: {len(events)}")
    
    print("\n[Event Table Structure]")
    print(f"Type of events[0]: {type(events[0])}")
    
    # We now know events[0] is an ndarray like ['19' 1 11 0 1]
    # Let's print the first 20 to understand the mapping manually
    print("\n[First 20 Events Raw Inspection]")
    for i in range(min(20, len(events))):
        print(f"Event {i:2d}: {events[i]}")
        
    # From the user's output: ['19' 1 11 0 1]
    # Let's assume:
    # [0] = type
    # [1] = latency (samples relative to epoch start?)
    # [2] = urevent?
    # [3] = duration?
    # [4] = epoch
    
    print("\n[Mapping Assumption]")
    print("Assuming index 0 = type, index 1 = latency, index 4 = epoch")
    
    # 1. FIND TRIAL 1 AUDIO MARKER
    audio_marker_val = None
    
    # We want Epoch 1. Let's find the audio marker for Epoch 1.
    for i, ev in enumerate(events):
        if len(ev) >= 5:
            t_str = str(ev[0]).strip()
            epoch_val = str(ev[4]).strip()
            
            if epoch_val == '1' and t_str.isdigit() and 11 <= int(t_str) <= 70:
                lat = ev[1]
                print(f"\n[Audio Marker Found] Index {i}")
                print(f"├── Type: {t_str}")
                print(f"├── Latency: {lat} (type: {type(lat)})")
                print(f"├── Epoch: {epoch_val}")
                audio_marker_val = t_str
                break
                
    if audio_marker_val is None:
        print("CRITICAL ERROR: Could not find audio marker for Trial 1.")
        return
        
    # 2. FIND BUTTON EVENTS FOR TRIAL 1
    print(f"\n[Button Events for Trial 1 (Epoch 1)]")
    
    trial_switches = []
    for ev in events:
        if len(ev) >= 5:
            t_str = str(ev[0]).strip()
            epoch_val = str(ev[4]).strip()
            
            if epoch_val == '1' and t_str in ['179', '184', '254', '255']:
                lat = float(ev[1])
                rel_sec = lat / 128.0
                print(f"├── Found {t_str} at latency {lat:.1f} ({rel_sec:.2f}s)")
                trial_switches.append((t_str, lat))
                
    if len(trial_switches) == 0:
        print("└── NO SWITCHES FOUND IN THIS EPOCH!")
    
    # 3. LOAD AUDIO
    print(f"\n[Audio]")
    audio_dir = '/kaggle/input/datasets/lokeshgile/aasd-audio-gammatones'
    npz_path = os.path.join(audio_dir, f"{int(audio_marker_val)}.npz")
    print(f"├── Loading WAV equivalents from: {npz_path}")
    
    if not os.path.exists(npz_path):
        print(f"CRITICAL ERROR: Audio file not found!")
        return
        
    audio_data = np.load(npz_path)
    env_l = audio_data['env_l']
    env_r = audio_data['env_r']
    print(f"├── Left envelope shape: {env_l.shape}")
    print(f"├── Right envelope shape: {env_r.shape}")
    
    # 4. CACHE / TARGET RECONSTRUCTION
    print(f"\n[Cache]")
    # Trial 1 is data_all[:, :, 0]
    trial_eeg = data_all[:, :, 0]
    print(f"├── EEG window shape: {trial_eeg.shape}")
    
    target_channels = ['T7', 'C2', 'FT8', 'P7', 'CPz', 'Fp1', 'TP8', 'C3']
    fallback_map = {'T7': 23, 'C2': 28, 'FT8': 22, 'P7': 41, 'CPz': 36, 'Fp1': 0, 'TP8': 40, 'C3': 25}
    sel_idx = [fallback_map[tc] for tc in target_channels]
    
    trial_eeg = trial_eeg - trial_eeg.mean(axis=0, keepdims=True)
    b, a = scipy.signal.butter(4, [1.0/64.0, 8.0/64.0], btype='band')
    trial_eeg_filt = scipy.signal.filtfilt(b, a, trial_eeg, axis=1)
    trial_eeg_64 = scipy.signal.resample_poly(trial_eeg_filt, 1, 2, axis=1)
    trial_eeg_8 = trial_eeg_64[sel_idx, :]
    
    print(f"├── Processed EEG shape (1-8Hz, 64Hz, 8ch): {trial_eeg_8.shape}")
    
    min_len = min(trial_eeg_8.shape[1], len(env_l))
    trial_eeg_8 = trial_eeg_8[:, :min_len]
    env_l = env_l[:min_len]
    env_r = env_r[:min_len]
    
    trial_eeg_8 = trial_eeg_8 - trial_eeg_8.mean(axis=1, keepdims=True)
    trial_eeg_8 = trial_eeg_8 / (trial_eeg_8.std(axis=1, keepdims=True) + 1e-12)
    
    env_l = (env_l - env_l.mean()) / (env_l.std() + 1e-12)
    env_r = (env_r - env_r.mean()) / (env_r.std() + 1e-12)
    
    # 5. MODEL EVALUATION
    print(f"\n[Model]")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AADConformer(in_channels=8).to(device)
    
    kul_path = '/kaggle/input/datasets/lowkieee/eeg-aad-conformer-seed1-checkpoints/checkpoints/seed_1/model_S1.pt'
    if os.path.exists(kul_path):
        ckpt = torch.load(kul_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt, strict=False)
        print(f"├── Loaded KUL weights (strict=False)")
    else:
        print(f"├── WARNING: KUL weights not found, using untrained model")
        
    model.eval()
    with torch.no_grad():
        eeg_tensor = torch.FloatTensor(trial_eeg_8).unsqueeze(0).to(device)
        pred, _ = model(eeg_tensor, return_features=True)
        pred = pred.squeeze().cpu().numpy()
        
    pred = pred[:min_len]
    
    corr_l = cross_corr(pred, env_l)
    corr_r = cross_corr(pred, env_r)
    
    print(f"├── Corr Left:  {corr_l:.4f}")
    print(f"├── Corr Right: {corr_r:.4f}")
    
    print(f"\n[Evaluation]")
    print(f"├── The model prefers: {'Left' if corr_l > corr_r else 'Right'} speaker")
    print(f"└── Why? Compare this preference to the Switch Events printed above!")

if __name__ == "__main__":
    main()
