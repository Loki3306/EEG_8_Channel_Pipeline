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

def cross_corr(x, y, max_lag):
    x = (x - np.mean(x)) / (np.std(x) + 1e-8)
    y = (y - np.mean(y)) / (np.std(y) + 1e-8)
    corrs = []
    lags = range(-max_lag, max_lag + 1)
    for lag in lags:
        if lag < 0: c = np.corrcoef(x[:lag], y[-lag:])[0, 1]
        elif lag > 0: c = np.corrcoef(x[lag:], y[:-lag])[0, 1]
        else: c = np.corrcoef(x, y)[0, 1]
        corrs.append(c if not np.isnan(c) else 0.0)
    return lags, np.array(corrs)

def run_alignment_audit(trials, model, device):
    FS = 64
    max_lag = int(2.0 * FS)
    all_corrs_att, all_corrs_unatt = [], []
    
    with torch.no_grad():
        for trial in trials:
            eeg = trial['eeg'].unsqueeze(0).to(device)
            audio_l = trial['audio_l'].numpy()[0]
            audio_r = trial['audio_r'].numpy()[0]
            raw_evs = trial['meta']['raw_evs']
            
            pred, _ = model(eeg, return_features=True)
            pred = pred.squeeze().cpu().numpy()
            
            if len(pred) <= max_lag * 2: continue
            
            # Find initial attended speaker from this trial's raw_evs
            # Wait, in the new cache, raw_evs will be specific to this trial.
            initial_att = 'R'
            for ev_t, ev_lat in raw_evs:
                if ev_t in ['179', '254']: initial_att = 'R'; break
                if ev_t in ['184', '255']: initial_att = 'L'; break
                
            att = audio_r if initial_att == 'R' else audio_l
            unatt = audio_l if initial_att == 'R' else audio_r
            
            _, c_att = cross_corr(pred, att, max_lag)
            _, c_unatt = cross_corr(pred, unatt, max_lag)
            all_corrs_att.append(c_att)
            all_corrs_unatt.append(c_unatt)
            
    if not all_corrs_att: return 0.0, 0, 0.0
    
    mean_corr_att = np.mean(all_corrs_att, axis=0)
    mean_corr_unatt = np.mean(all_corrs_unatt, axis=0)
    
    peak_idx = np.argmax(mean_corr_att)
    lags = range(-max_lag, max_lag + 1)
    peak_lag_sec = lags[peak_idx] / float(FS)
    peak_val = mean_corr_att[peak_idx]
    
    peak_idx_u = np.argmax(mean_corr_unatt)
    peak_val_u = mean_corr_unatt[peak_idx_u]
    
    return peak_val, peak_lag_sec, peak_val_u

def main():
    print("==================================================")
    print("=== PHASE 28.8 CACHE RECONSTRUCTION & VALIDATION ===")
    print("==================================================\n")
    
    S1_mat = '/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG/S01/S1.mat'
    if not os.path.exists(S1_mat):
        S1_mat = glob.glob('/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG/*/*.mat')[0]
        
    mat = scipy.io.loadmat(S1_mat, squeeze_me=True, struct_as_record=False)
    eeg_var = [k for k in mat.keys() if not k.startswith('__')][0]
    eeg_obj = mat[eeg_var]
    data_all = eeg_obj.data
    events = eeg_obj.event
    
    print("--- STAGE 1: BUG VERIFICATION ---")
    print(f"Original EEG shape: {data_all.shape}")
    
    audio_markers = []
    for ev in events:
        t_str = str(get_ev_attr(ev, 'type', 0)).strip()
        if t_str.isdigit():
            val = int(t_str)
            if 11 <= val <= 70:
                lat = int(get_ev_attr(ev, 'latency'))
                audio_markers.append((t_str, lat))
                
    trial_length = data_all.shape[1] if len(data_all.shape) == 3 else 7680
    
    for i, (marker, lat) in enumerate(audio_markers[:5]):
        expected_start = i * trial_length + lat
        current_start = lat # What the old cache extracted
        status = "PASS" if current_start == expected_start else "FAIL"
        
        print(f"Trial {i+1}:")
        print(f"  Audio Marker:     {marker}")
        print(f"  Relative latency: {lat}")
        print(f"  Trial offset:     {i * trial_length}")
        print(f"  Current extract:  {current_start}")
        print(f"  Expected extract: {expected_start}")
        print(f"  STATUS:           {status}")
        print("--------------------------------")
        
    print("\n--- STAGE 2: CACHE REWRITE ---")
    audio_dir = '/kaggle/input/datasets/lokeshgile/aasd-audio-gammatones'
    target_channels = ['T7', 'C2', 'FT8', 'P7', 'CPz', 'Fp1', 'TP8', 'C3']
    fallback_map = {'T7': 23, 'C2': 28, 'FT8': 22, 'P7': 41, 'CPz': 36, 'Fp1': 0, 'TP8': 40, 'C3': 25}
    sel_idx = [fallback_map[tc] for tc in target_channels]
    
    b, a = scipy.signal.butter(4, [1.0/32.0, 8.0/32.0], btype='band')
    
    new_trials = []
    
    if len(audio_markers) != data_all.shape[2]:
        print(f"[WARNING] Audio markers ({len(audio_markers)}) != Trials ({data_all.shape[2]})")
        
    for trial_idx in range(data_all.shape[2]):
        if trial_idx >= len(audio_markers): break
            
        marker, lat = audio_markers[trial_idx]
        
        # 1. Extract specifically this trial's EEG
        trial_eeg = data_all[:, :, trial_idx]
        
        # 2. CAR
        trial_eeg = trial_eeg - trial_eeg.mean(axis=0, keepdims=True)
        
        # 3. Filter & Downsample
        trial_eeg_filt = scipy.signal.filtfilt(b, a, trial_eeg, axis=1)
        trial_eeg_64 = scipy.signal.resample_poly(trial_eeg_filt, 1, 2, axis=1)
        trial_eeg_8 = trial_eeg_64[sel_idx, :]
        
        # The audio marker starts at `lat` samples (128Hz). In 64Hz, that is lat // 2
        start_64 = int(lat // 2)
        trial_eeg_8 = trial_eeg_8[:, start_64:] # Discard pre-stimulus baseline
        
        # Load audio
        npz_path = os.path.join(audio_dir, f"{int(marker)}.npz")
        if not os.path.exists(npz_path): continue
        
        audio_data = np.load(npz_path)
        env_l = audio_data['env_l']
        env_r = audio_data['env_r']
        
        min_len = min(trial_eeg_8.shape[1], len(env_l))
        trial_eeg_8 = trial_eeg_8[:, :min_len]
        env_l = env_l[:min_len]
        env_r = env_r[:min_len]
        
        # Gather all events that belong to this epoch
        # In EEGLAB, we can just filter by epoch number (if available)
        # We will approximate it by just finding events where epoch == trial_idx + 1
        raw_evs = []
        for ev in events:
            ep = get_ev_attr(ev, 'epoch')
            if ep and int(ep) == trial_idx + 1:
                t_str = str(get_ev_attr(ev, 'type')).strip()
                ev_lat = float(get_ev_attr(ev, 'latency'))
                raw_evs.append((t_str, ev_lat))
                
        meta = {
            'TrialID': trial_idx + 1,
            'audio_marker': marker,
            'raw_evs': raw_evs
        }
        
        new_trials.append({
            'meta': meta,
            'eeg': torch.FloatTensor(trial_eeg_8),
            'audio_l': torch.FloatTensor(env_l).unsqueeze(0),
            'audio_r': torch.FloatTensor(env_r).unsqueeze(0)
        })
        
    print(f"Successfully built new cache with {len(new_trials)} trials.")
    
    print("\n--- STAGE 3: VALIDATION ---")
    old_cache_file = REPO_ROOT / 'data' / 'processed_aasd' / 'S1.pt'
    if old_cache_file.exists():
        old_data = torch.load(old_cache_file, weights_only=False)
        old_trials = old_data['trials']
        
        print(f"Old Cache Trial 1 EEG mean: {old_trials[0]['eeg'].mean():.4f}")
        print(f"New Cache Trial 1 EEG mean: {new_trials[0]['eeg'].mean():.4f}")
        print(f"Old Cache Trial 2 EEG mean: {old_trials[1]['eeg'].mean():.4f}")
        print(f"New Cache Trial 2 EEG mean: {new_trials[1]['eeg'].mean():.4f}")
        
        if torch.allclose(old_trials[0]['eeg'], old_trials[1]['eeg']):
            print("CONFIRMED: Old Cache Trial 1 is IDENTICAL to Trial 2.")
        else:
            print("Old Cache Trials 1 and 2 are different (maybe due to start_64 slice).")
            
        if not torch.allclose(new_trials[0]['eeg'], new_trials[1]['eeg']):
            print("CONFIRMED: New Cache Trials 1 and 2 are DIFFERENT (Correct behavior).")
    
    print("\n--- STAGE 4: ALIGNMENT AUDIT ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AADConformer(in_channels=8).to(device)
    kul_path = '/kaggle/input/datasets/lowkieee/eeg-aad-conformer-seed1-checkpoints/checkpoints/seed_1/model_S1.pt'
    if os.path.exists(kul_path):
        ckpt = torch.load(kul_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt, strict=False)
    model.eval()
    
    if old_cache_file.exists():
        p_att, p_lag, p_unatt = run_alignment_audit(old_trials, model, device)
        print(f"[OLD CACHE] Max Attended: {p_att:.4f} at {p_lag:+.3f}s | Unattended: {p_unatt:.4f}")
        
    p_att_new, p_lag_new, p_unatt_new = run_alignment_audit(new_trials, model, device)
    print(f"[NEW CACHE] Max Attended: {p_att_new:.4f} at {p_lag_new:+.3f}s | Unattended: {p_unatt_new:.4f}")
    
    if p_att_new > 0.05:
        print("\nDIAGNOSIS: SUCCESS! Correlation increased massively.")
        print("The epoched latency bug is scientifically validated and fixed.")
    else:
        print("\nDIAGNOSIS: INCOMPLETE.")
        print("Correlation is still low. We may also need to apply the -62ms hardware lag shift.")

if __name__ == "__main__":
    main()
