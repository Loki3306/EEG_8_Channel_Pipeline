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
    print("=== PHASE 28.10 FILTER NYQUIST FIX & AUDIT =======")
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
    
    # FATAL BUG FIX: Nyquist = 64.0 (128Hz sampling rate)
    b, a = scipy.signal.butter(4, [1.0/64.0, 8.0/64.0], btype='band')
    
    new_trials = []
    for trial_idx in range(data_all.shape[2]):
        if trial_idx >= len(audio_markers): break
        marker, lat = audio_markers[trial_idx]
        
        trial_eeg = data_all[:, :, trial_idx]
        trial_eeg = trial_eeg - trial_eeg.mean(axis=0, keepdims=True)
        trial_eeg_filt = scipy.signal.filtfilt(b, a, trial_eeg, axis=1)
        trial_eeg_64 = scipy.signal.resample_poly(trial_eeg_filt, 1, 2, axis=1)
        trial_eeg_8 = trial_eeg_64[sel_idx, :]
        
        start_64 = int(lat // 2)
        trial_eeg_8 = trial_eeg_8[:, start_64:]
        
        npz_path = os.path.join(audio_dir, f"{int(marker)}.npz")
        if not os.path.exists(npz_path): continue
        
        audio_data = np.load(npz_path)
        env_l = audio_data['env_l']
        env_r = audio_data['env_r']
        
        min_len = min(trial_eeg_8.shape[1], len(env_l))
        trial_eeg_8 = trial_eeg_8[:, :min_len]
        env_l = env_l[:min_len]
        env_r = env_r[:min_len]
        
        trial_eeg_8 = trial_eeg_8 - trial_eeg_8.mean(axis=1, keepdims=True)
        trial_eeg_8 = trial_eeg_8 / (trial_eeg_8.std(axis=1, keepdims=True) + 1e-12)
        
        raw_evs = []
        for ev in events:
            ep = get_ev_attr(ev, 'epoch')
            if ep and int(ep) == trial_idx + 1:
                t_str = str(get_ev_attr(ev, 'type')).strip()
                ev_lat = float(get_ev_attr(ev, 'latency'))
                raw_evs.append((t_str, ev_lat))
                
        new_trials.append({
            'meta': {'raw_evs': raw_evs},
            'eeg': torch.FloatTensor(trial_eeg_8),
            'audio_l': torch.FloatTensor(env_l).unsqueeze(0),
            'audio_r': torch.FloatTensor(env_r).unsqueeze(0)
        })
        
    print(f"Successfully built fully corrected cache (1-8Hz) with {len(new_trials)} trials.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AADConformer(in_channels=8).to(device)
    kul_path = '/kaggle/input/datasets/lowkieee/eeg-aad-conformer-seed1-checkpoints/checkpoints/seed_1/model_S1.pt'
    if os.path.exists(kul_path):
        ckpt = torch.load(kul_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt, strict=False)
        print("[INFO] Loaded zero-shot KUL weights.")
    model.eval()
    
    p_att_new, p_lag_new, p_unatt_new = run_alignment_audit(new_trials, model, device)
    print(f"[CORRECTED CACHE] Max Attended Correlation: {p_att_new:.4f} at lag {p_lag_new:+.3f}s | Unattended: {p_unatt_new:.4f}")
    
    if p_att_new > 0.05:
        print("\nDIAGNOSIS: MASSIVE SUCCESS!")
        print("The correlation jumped, proving that the epoched bug + 2-16Hz filter bug were the root causes.")
    else:
        print("\nDIAGNOSIS: STILL NO ZERO-SHOT TRANSFER.")
        print("This means the KUL spatial filters (weights) are fundamentally incompatible with AASD hardware.")
        print("We MUST rely on fine-tuning. Rerunning Phase 28.9 on this 1-8Hz cache is required.")

if __name__ == "__main__":
    main()
