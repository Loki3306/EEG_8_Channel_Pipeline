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

def get_ev_attr(e, attr_name, array_idx=0):
    try:
        if hasattr(e, attr_name): return getattr(e, attr_name)
        if hasattr(e.flat[0], attr_name): return getattr(e.flat[0], attr_name)
        return e[array_idx]
    except: return ''

def cross_corr(x, y):
    if len(x) == 0 or len(y) == 0: return 0.0
    x = (x - np.mean(x)) / (np.std(x) + 1e-8)
    y = (y - np.mean(y)) / (np.std(y) + 1e-8)
    c = np.corrcoef(x, y)[0, 1]
    return c if not np.isnan(c) else 0.0

def main():
    print("==================================================")
    print("=== PHASE 28.14 LABEL CONSISTENCY AUDIT =========")
    print("==================================================\n")
    
    S1_mat = '/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG/S01/S1.mat'
    if not os.path.exists(S1_mat):
        S1_mat = glob.glob('/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG/*/*.mat')[0]
        
    mat = scipy.io.loadmat(S1_mat, squeeze_me=True, struct_as_record=False)
    eeg_var = [k for k in mat.keys() if not k.startswith('__')][0]
    data_all = mat[eeg_var].data
    events = mat[eeg_var].event
    
    audio_markers = []
    for ev in events:
        t_str = str(get_ev_attr(ev, 'type', 0)).strip()
        if t_str.isdigit():
            val = int(t_str)
            if 11 <= val <= 70:
                lat = int(get_ev_attr(ev, 'latency'))
                audio_markers.append((t_str, lat))
                
    audio_dir = '/kaggle/input/datasets/lokeshgile/aasd-audio-gammatones'
    target_channels = ['T7', 'C2', 'FT8', 'P7', 'CPz', 'Fp1', 'TP8', 'C3']
    fallback_map = {'T7': 23, 'C2': 28, 'FT8': 22, 'P7': 41, 'CPz': 36, 'Fp1': 0, 'TP8': 40, 'C3': 25}
    sel_idx = [fallback_map[tc] for tc in target_channels]
    
    b, a = scipy.signal.butter(4, [1.0/64.0, 8.0/64.0], btype='band')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AADConformer(in_channels=8).to(device)
    kul_path = '/kaggle/input/datasets/lowkieee/eeg-aad-conformer-seed1-checkpoints/checkpoints/seed_1/model_S1.pt'
    if os.path.exists(kul_path):
        ckpt = torch.load(kul_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt, strict=False)
        
    model.eval()
    
    train_trials = int(data_all.shape[2] * 0.8)
    
    for trial_idx in range(data_all.shape[2]):
        if trial_idx >= len(audio_markers): break
        
        split_type = "TRAIN" if trial_idx < train_trials else "TEST"
        print(f"\n==================================================")
        print(f"=== TRIAL {trial_idx+1} ({split_type}) ===")
        
        marker, lat = audio_markers[trial_idx]
        print(f"Audio marker event: {marker} at latency {lat}")
        
        npz_path = os.path.join(audio_dir, f"{int(marker)}.npz")
        if not os.path.exists(npz_path):
            print(f"ERROR: WAV filename loaded: {npz_path} NOT FOUND")
            continue
            
        print(f"WAV filename loaded: {npz_path}")
        print("audio_a source: env_l (from npz)")
        print("audio_b source: env_r (from npz)")
        
        raw_evs = []
        for ev in events:
            ep = get_ev_attr(ev, 'epoch')
            if ep and int(ep) == trial_idx + 1:
                t_str = str(get_ev_attr(ev, 'type')).strip()
                ev_lat = float(get_ev_attr(ev, 'latency'))
                raw_evs.append((t_str, ev_lat))
                
        initial_att = 'Unknown'
        for ev_t, ev_lat in raw_evs:
            if ev_t in ['179', '254']: initial_att = 'R (179/254)'; break
            if ev_t in ['184', '255']: initial_att = 'L (184/255)'; break
            
        print(f"Initial attended state: {initial_att}")
        print("Button timeline:")
        for ev_t, ev_lat in raw_evs:
            if ev_t in ['179', '184', '254', '255']:
                t_sec = (ev_lat - lat) / 128.0
                print(f"  {t_sec:+.2f}s : {ev_t} -> {'Right' if ev_t in ['179', '254'] else 'Left'}")
                
        # We will use the constant label logic for correlation test since the dynamic one failed
        constant_att_str = 'R' if 'R' in initial_att else ('L' if 'L' in initial_att else 'Unknown')
        print(f"Current target (Constant assumption): audio_{'b (R)' if constant_att_str == 'R' else 'a (L)'} = attended")
        
        # Build 1-8Hz EEG
        trial_eeg = data_all[:, :, trial_idx]
        trial_eeg = trial_eeg - trial_eeg.mean(axis=0, keepdims=True)
        trial_eeg_filt = scipy.signal.filtfilt(b, a, trial_eeg, axis=1)
        trial_eeg_64 = scipy.signal.resample_poly(trial_eeg_filt, 1, 2, axis=1)
        trial_eeg_8 = trial_eeg_64[sel_idx, :]
        
        start_64 = int(lat // 2)
        trial_eeg_8 = trial_eeg_8[:, start_64:]
        
        audio_data = np.load(npz_path)
        env_l = audio_data['env_l']
        env_r = audio_data['env_r']
        
        min_len = min(trial_eeg_8.shape[1], len(env_l))
        trial_eeg_8 = trial_eeg_8[:, :min_len]
        env_l = env_l[:min_len]
        env_r = env_r[:min_len]
        
        trial_eeg_8 = trial_eeg_8 - trial_eeg_8.mean(axis=1, keepdims=True)
        trial_eeg_8 = trial_eeg_8 / (trial_eeg_8.std(axis=1, keepdims=True) + 1e-12)
        
        env_l = (env_l - env_l.mean()) / (env_l.std() + 1e-12)
        env_r = (env_r - env_r.mean()) / (env_r.std() + 1e-12)
        
        if constant_att_str == 'R':
            att = env_r
            unatt = env_l
        else:
            att = env_l
            unatt = env_r
            
        with torch.no_grad():
            eeg_tensor = torch.FloatTensor(trial_eeg_8).unsqueeze(0).to(device)
            pred, _ = model(eeg_tensor, return_features=True)
            pred = pred.squeeze().cpu().numpy()
            pred = pred[:min_len]
            
            corr_a = cross_corr(pred, env_l) # Left
            corr_b = cross_corr(pred, env_r) # Right
            
            corr_att = cross_corr(pred, att)
            corr_unatt = cross_corr(pred, unatt)
            
        print("Correlation:")
        print(f"  A (Left)  = {corr_a:.4f}")
        print(f"  B (Right) = {corr_b:.4f}")
        
        pred_state = "Left (A)" if corr_a > corr_b else "Right (B)"
        ground_truth = "Right (B)" if constant_att_str == 'R' else "Left (A)"
        
        print(f"Ground truth state: {ground_truth}")
        print(f"Predicted state:    {pred_state}")
        
        match = "YES" if pred_state == ground_truth else "NO"
        print(f"Prediction matches reconstructed ground truth: {match}")
        
        # We stop early just to limit the massive stdout. Let's do first 3 TRAIN and first 3 TEST.
        if trial_idx == 2:
            print("\n... skipping to TEST trials ...\n")
        elif trial_idx > 2 and trial_idx < train_trials:
            continue
        if trial_idx == train_trials + 2:
            break

if __name__ == "__main__":
    main()
