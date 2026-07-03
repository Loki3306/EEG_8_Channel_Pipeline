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

def get_raw_field(ev, field_name):
    # ABSOLUTELY NO ABSTRACTIONS OR FALLBACKS.
    # We inspect EXACTLY what scipy.io gives us.
    if hasattr(ev, field_name):
        val = getattr(ev, field_name)
    elif hasattr(ev, 'dtype') and field_name in ev.dtype.names:
        val = ev[field_name]
    elif isinstance(ev, np.void) and field_name in ev.dtype.names:
        val = ev[field_name]
    else:
        # If it's a mat_struct array that got squeezed weirdly
        try:
            val = ev.__dict__[field_name]
        except:
            return None
            
    # scipy.io often wraps scalars in 1D arrays or nested arrays
    if isinstance(val, np.ndarray):
        if val.size == 1:
            val = val.flat[0]
        elif val.size == 0:
            val = None
            
    # Sometimes it's a string, sometimes a number. Let's return as string if possible, or int/float
    if isinstance(val, str):
        return val.strip()
    return val

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
    if hasattr(events[0], 'dtype'):
        print(f"dtype names: {events[0].dtype.names}")
    elif hasattr(events[0], '__dict__'):
        print(f"__dict__ keys: {list(events[0].__dict__.keys())}")
        
    print("\nRaw inspection of events[0]:")
    print(events[0])
    
    # 1. FIND TRIAL 1 AUDIO MARKER
    audio_marker_val = None
    trial_start_latency = None
    
    for i, ev in enumerate(events):
        t = get_raw_field(ev, 'type')
        if t is not None:
            t_str = str(t).strip()
            if t_str.isdigit():
                val = int(t_str)
                if 11 <= val <= 70:
                    lat = get_raw_field(ev, 'latency')
                    print(f"\n[Audio Marker Found] Index {i}")
                    print(f"├── Type: {t_str}")
                    print(f"├── Raw Latency: {lat} (type: {type(lat)})")
                    audio_marker_val = t_str
                    trial_start_latency = float(lat)
                    break
                    
    if trial_start_latency is None:
        print("CRITICAL ERROR: Could not find audio marker for Trial 1.")
        return
        
    # 2. FIND BUTTON EVENTS FOR TRIAL 1
    # We look for 179, 184, 254, 255 that occur AFTER trial_start_latency
    # and BEFORE the next audio marker (or trial_start_latency + 7680)
    # Wait, the data matrix is (62, 7680, 60). 7680 samples = 60 seconds @ 128Hz.
    print(f"\n[Button Events for Trial 1]")
    print(f"├── Searching latencies between {trial_start_latency} and {trial_start_latency + 7680}")
    
    trial_switches = []
    for ev in events:
        t = get_raw_field(ev, 'type')
        if t is not None:
            t_str = str(t).strip()
            if t_str in ['179', '184', '254', '255']:
                lat = float(get_raw_field(ev, 'latency') or 0)
                if trial_start_latency <= lat < trial_start_latency + 7680:
                    rel_lat = lat - trial_start_latency
                    rel_sec = rel_lat / 128.0
                    print(f"├── Found {t_str} at absolute {lat:.1f} (relative {rel_lat:.1f} samples / {rel_sec:.2f}s)")
                    trial_switches.append((t_str, rel_lat))
                    
    if len(trial_switches) == 0:
        print("└── NO SWITCHES FOUND IN THIS WINDOW!")
    
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
    # We know data_all is (62, 7680, 60). Trial 1 is data_all[:, :, 0]
    # But wait, did they align it perfectly so data_all[:, :, 0] matches the latencies?
    # In EEGLAB, epoched data puts the epoch center at 0 latency usually, but let's assume it's pre-sliced.
    trial_eeg = data_all[:, :, 0]
    print(f"├── EEG window shape: {trial_eeg.shape}")
    
    # Let's extract the correct channels and filter
    target_channels = ['T7', 'C2', 'FT8', 'P7', 'CPz', 'Fp1', 'TP8', 'C3']
    fallback_map = {'T7': 23, 'C2': 28, 'FT8': 22, 'P7': 41, 'CPz': 36, 'Fp1': 0, 'TP8': 40, 'C3': 25}
    sel_idx = [fallback_map[tc] for tc in target_channels]
    
    trial_eeg = trial_eeg - trial_eeg.mean(axis=0, keepdims=True)
    b, a = scipy.signal.butter(4, [1.0/64.0, 8.0/64.0], btype='band')
    trial_eeg_filt = scipy.signal.filtfilt(b, a, trial_eeg, axis=1)
    trial_eeg_64 = scipy.signal.resample_poly(trial_eeg_filt, 1, 2, axis=1)
    trial_eeg_8 = trial_eeg_64[sel_idx, :]
    
    print(f"├── Processed EEG shape (1-8Hz, 64Hz, 8ch): {trial_eeg_8.shape}")
    
    # Ensure lengths match
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
