import scipy.io
import os
import glob
import numpy as np
import sys
import torch
import torch.nn as nn
import scipy.signal as signal
from sklearn.metrics import roc_curve, auc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.aad_conformer import AADConformer

def norm_env(env):
    env = env - env.mean(axis=1, keepdims=True)
    env = env / (env.std(axis=1, keepdims=True) + 1e-12)
    return env

def safe_corr_np(x, y, eps=1e-8):
    x_mean = x.mean(axis=1, keepdims=True)
    y_mean = y.mean(axis=1, keepdims=True)
    num = np.sum((x - x_mean) * (y - y_mean), axis=1)
    den = np.sqrt(np.sum((x - x_mean)**2, axis=1) * np.sum((y - y_mean)**2, axis=1))
    return num / (den + eps)

def main():
    print("====================================================")
    print("PHASE 25A.8: CHANNEL & ALIGNMENT FORENSIC AUDIT")
    print("====================================================")

    data_dir = "/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG"
    audio_dir = "/kaggle/input/datasets/lokeshgile/aasd-audio-gammatones"
    ckpt_path = "/kaggle/input/datasets/lowkieee/eeg-aad-conformer-seed1-checkpoints/checkpoints/seed_1/model_S1.pt"

    mat_files = glob.glob(os.path.join(data_dir, '*', '*.mat'))
    if not mat_files:
        print("[FAIL] No mat files found.")
        return
        
    s18_path = next((mf for mf in mat_files if 'S18' in mf), mat_files[0])
    subj = os.path.basename(os.path.dirname(s18_path))
    print(f"[INFO] Analyzing Subject: {subj}")
    
    mat = scipy.io.loadmat(s18_path, squeeze_me=True, struct_as_record=False)
    eeg_var = [k for k in mat.keys() if not k.startswith('__')][0]
    data = mat[eeg_var].data
    events = mat[eeg_var].event
    
    print(f"[INFO] Data shape: {data.shape}")
    
    # 1. CHANNEL AUDIT
    chan_names = []
    try:
        chanlocs = mat[eeg_var].chanlocs
        if isinstance(chanlocs, np.ndarray):
            for c in chanlocs:
                lbl = getattr(c, 'labels', '')
                if isinstance(lbl, (list, np.ndarray)) and len(lbl) > 0:
                    lbl = lbl[0]
                chan_names.append(str(lbl).strip().upper())
        print(f"[INFO] Successfully loaded {len(chan_names)} channel names from chanlocs.")
        print(f"Channels: {chan_names[:10]} ... {chan_names[-10:]}")
    except Exception as e:
        print(f"[WARN] Failed to load chanlocs: {e}")
        
    target_channels = ['T7', 'C2', 'FT8', 'P7', 'CPz', 'Fp1', 'TP8', 'C3']
    sel_idx = []
    
    if len(chan_names) > 0:
        for tc in target_channels:
            if tc.upper() in chan_names:
                sel_idx.append(chan_names.index(tc.upper()))
            else:
                print(f"[FAIL] Target channel {tc} missing!")
    else:
        print("[WARN] Using Fallback Map!")
        # The AASD dataset was recorded using a Neuroscan 64-channel system.
        # Standard Neuroscan 64 channel order (0-indexed):
        # FP1(0), FPZ(1), FP2(2), AF3(3), AF4(4), F7(5), F5(6), F3(7), F1(8), FZ(9),
        # F2(10), F4(11), F6(12), F8(13), FT7(14), FC5(15), FC3(16), FC1(17), FCZ(18), FC2(19),
        # FC4(20), FC6(21), FT8(22), T7(23), C5(24), C3(25), C1(26), CZ(27), C2(28), C4(29),
        # C6(30), T8(31), TP7(32), CP5(33), CP3(34), CP1(35), CPZ(36), CP2(37), CP4(38), CP6(39),
        # TP8(40), P7(41), P5(42), P3(43), P1(44), PZ(45), P2(46), P4(47), P6(48), P8(49),
        # PO7(50), PO5(51), PO3(52), POZ(53), PO4(54), PO6(55), PO8(56), CB1(57), O1(58), OZ(59),
        # O2(60), CB2(61)
        fallback_map = {'T7': 23, 'C2': 28, 'FT8': 22, 'P7': 41, 'CPz': 36, 'Fp1': 0, 'TP8': 40, 'C3': 25}
        sel_idx = [fallback_map[tc] for tc in target_channels]
        
    print(f"[INFO] Selected Indices: {sel_idx}")
    
    # 2. MODEL PREDICTION ALIGNMENT AUDIT (TRIAL 1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AADConformer(in_channels=8).to(device)
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict.get('model_state_dict', state_dict), strict=False)
    model.eval()

    # Find trial 1 events
    epoch_events = []
    for i in range(events.shape[0]):
        ev = events[i] if events.ndim > 1 else events
        if events.ndim > 1 and len(ev) >= 5:
            ep, typ, lat = int(ev[4]), str(ev[0]).strip(), float(ev[1])
        else:
            ep = int(getattr(ev, 'epoch', 0))
            typ = str(getattr(ev, 'type', '')).strip()
            lat = float(getattr(ev, 'latency', 0))
            
        if ep == 1:
            epoch_events.append((typ, lat))
            
    epoch_events.sort(key=lambda x: x[1])
    
    audio_marker, trial_start_samples = None, 0
    switch_events = []
    for ev_t, ev_lat in epoch_events:
        if ev_t not in ['179', '184', '254', '255'] and audio_marker is None:
            audio_marker = ev_t
            trial_start_samples = ev_lat
        elif ev_t in ['179', '184']:
            switch_events.append((ev_t, ev_lat))
            
    if not audio_marker:
        print("[FAIL] No audio marker found in trial 1.")
        return
        
    marker_id = int(audio_marker)
    npz_path = os.path.join(audio_dir, f"{marker_id}.npz")
    if not os.path.exists(npz_path):
        print(f"[FAIL] Audio {npz_path} missing.")
        return
        
    audio_data = np.load(npz_path)
    env_l_1d, env_r_1d = audio_data['env_l'], audio_data['env_r']
    
    print(f"\n[INFO] Trial 1 - Audio Marker: {marker_id}")
    switch_times_sec = [(lat - trial_start_samples) / 128.0 for t, lat in switch_events if (lat - trial_start_samples)/128.0 > 0]
    types_B = ['R' if t == '179' else 'L' for t, lat in switch_events if (lat - trial_start_samples)/128.0 > 0]
    
    print(f"[INFO] Switches (Hypothesis B): {list(zip(switch_times_sec, types_B))}")
    
    # Preprocess EEG
    nyq = 128 / 2
    b, a = signal.butter(4, [1.0/nyq, 8.0/nyq], btype='band')
    eeg_filt = signal.filtfilt(b, a, data[:, :, 0], axis=1) # Trial 1
    eeg_8 = eeg_filt[sel_idx, :]
    
    import math
    g = math.gcd(64, 128)
    eeg_64 = signal.resample_poly(eeg_8, 64 // g, 128 // g, axis=1)
    eeg_norm = norm_env(eeg_64)
    
    # Ground Truth Generation
    current_state = 1 if types_B[0] == 'R' else -1
    t_array = np.arange(0, min(eeg_norm.shape[1], len(env_l_1d)) - 128, 64) / 64.0 + 1.0 # Centers
    gt_B = np.zeros(len(t_array))
    for i, t in enumerate(t_array):
        state = current_state
        for st, s_type in zip(switch_times_sec, types_B):
            if t >= st:
                state = 1 if s_type == 'L' else -1
        gt_B[i] = state
        
    # Inference
    margins, predictions = [], []
    c_l_all, c_r_all = [], []
    for start in range(0, min(eeg_norm.shape[1], len(env_l_1d)) - 128, 64):
        win_eeg = eeg_norm[:, start:start+128]
        w_eeg_mean = win_eeg.mean(axis=1, keepdims=True)
        w_eeg_std = win_eeg.std(axis=1, keepdims=True) + 1e-8
        win_eeg_norm = (win_eeg - w_eeg_mean) / w_eeg_std
        
        win_l = env_l_1d[start:start+128]
        win_r = env_r_1d[start:start+128]
        win_l_norm = (win_l - win_l.mean()) / (win_l.std() + 1e-8)
        win_r_norm = (win_r - win_r.mean()) / (win_r.std() + 1e-8)
        
        eeg_t = torch.tensor(win_eeg_norm[np.newaxis, ...], dtype=torch.float32).to(device)
        with torch.no_grad():
            out, _ = model(eeg_t, return_features=True)
            pred_env = out.squeeze(1).cpu().numpy()
            
        c_l = safe_corr_np(pred_env, win_l_norm[np.newaxis, ...])[0]
        c_r = safe_corr_np(pred_env, win_r_norm[np.newaxis, ...])[0]
        margin = c_l - c_r
        
        c_l_all.append(c_l)
        c_r_all.append(c_r)
        margins.append(margin)
        predictions.append(1 if margin > 0 else -1)
        
    print("\n| Time (s) | GT (Hyp B) | L-Corr | R-Corr | Margin | Pred (1=L) | Match? |")
    print("|----------|------------|--------|--------|--------|------------|--------|")
    for i in range(25): # Print first 25 windows (approx 25s)
        t = t_array[i]
        gt = int(gt_B[i])
        m = margins[i]
        p = predictions[i]
        c_l = c_l_all[i]
        c_r = c_r_all[i]
        match = "YES" if p == gt else "NO"
        print(f"| {t:8.2f} | {gt:10d} | {c_l:6.3f} | {c_r:6.3f} | {m:6.3f} | {p:10d} | {match:6s} |")

    # Analyze margins
    print(f"\n[INFO] Mean Margin: {np.mean(margins):.4f} | Std Margin: {np.std(margins):.4f}")
    
if __name__ == "__main__":
    main()
