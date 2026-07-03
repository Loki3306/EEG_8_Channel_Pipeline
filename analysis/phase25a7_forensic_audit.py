import torch
import numpy as np
import scipy.io
import scipy.signal as signal
import os
import sys

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
    print("PHASE 25A.7: WINDOW-BY-WINDOW FORENSIC AUDIT")
    print("====================================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AADConformer(in_channels=8)
    ckpt_path = "/kaggle/input/datasets/lowkieee/eeg-aad-conformer-seed1-checkpoints/checkpoints/seed_1/model_S1.pt"
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location='cpu')
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    mat_path = "/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG/S18/S18.mat"
    if not os.path.exists(mat_path):
        print(f"[FAIL] Missing {mat_path}.")
        sys.exit(1)
        
    mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    eeg_var = [k for k in mat.keys() if not k.startswith('__')][0]
    eeg_data = mat[eeg_var].data
    events = mat[eeg_var].event
    
    # Extract Epoch 1 (Trial 1)
    trial_eeg_128 = eeg_data[:, :, 0]
    
    # Filter 1-8Hz
    nyq = 128 / 2
    b, a = signal.butter(4, [1.0/nyq, 8.0/nyq], btype='band')
    eeg_filt = signal.filtfilt(b, a, trial_eeg_128, axis=1)
    
    # Channel mapping (BioSemi 64 to KUL)
    # T7, C2, FT8, P7, CPz, Fp1, TP8, C3
    fallback_map = {'T7': 14, 'C2': 43, 'FT8': 39, 'P7': 22, 'CPz': 31, 'Fp1': 0, 'TP8': 47, 'C3': 12}
    sel_idx = [fallback_map[tc] for tc in ['T7', 'C2', 'FT8', 'P7', 'CPz', 'Fp1', 'TP8', 'C3']]
    eeg_8 = eeg_filt[sel_idx, :]
    
    # Downsample to 64Hz
    import math
    g = math.gcd(64, 128)
    eeg_64 = signal.resample_poly(eeg_8, 64 // g, 128 // g, axis=1)
    eeg_norm = norm_env(eeg_64)
    
    epoch_events = []
    for i in range(events.shape[0]):
        ev = events[i] if events.ndim > 1 else events
        
        if events.ndim > 1 and len(ev) >= 5:
            ep = int(ev[4])
            typ = str(ev[0]).strip()
            lat = float(ev[1])
        else:
            ep = int(getattr(ev, 'epoch', 0))
            typ = str(getattr(ev, 'type', '')).strip()
            lat = float(getattr(ev, 'latency', 0))
            
        if ep == 1:
            epoch_events.append((typ, lat))
            
    epoch_events.sort(key=lambda x: x[1])
    
    audio_marker = None
    switch_events = []
    for ev_t, ev_lat in epoch_events:
        if ev_t not in ['179', '184', '254', '255']:
            audio_marker = ev_t
        elif ev_t in ['179', '184']:
            switch_events.append((ev_t, ev_lat))
            
    # Note: For epoched data, the audio starts at t=0 of the epoch.
    # The events are already aligned with the start of the epoch.
    switch_times_sec = []
    types_B = [] # Hypothesis B: 179=R, 184=L
    
    for t, lat in switch_events:
        # the latency is continuous time, but wait...
        # Let's see what the continuous latency was.
        # Actually in EEGLAB, if data is epoched, `latency` might be in continuous time.
        # Let's assume the first event (e.g. audio marker) is the start of the epoch!
        t_sec = lat / 128.0
        switch_times_sec.append(t_sec)
        types_B.append('R' if t == '179' else 'L')
        
    print(f"[INFO] Audio Marker: {audio_marker}")
    print(f"[INFO] Switch Events (Hypothesis B): {list(zip(switch_times_sec, types_B))}")
    
    # But wait! If latency is continuous, then `lat / 128.0` is huge!
    # Let's normalize it to the first event in the epoch.
    first_lat = epoch_events[0][1]
    switch_times_sec = [(lat - first_lat) / 128.0 for t, lat in switch_events]
    print(f"[INFO] Normalized Switch Times: {switch_times_sec}")
    
    # Determine initial state (Hypothesis B)
    if len(types_B) > 0:
        current_state = 'L' if types_B[0] == 'R' else 'R'
    else:
        current_state = 'L'
        
    # Load audio
    cache_path_1 = f"/kaggle/working/results/phase25a5/audio_cache/{int(audio_marker)}.npz"
    cache_path_2 = f"/kaggle/input/datasets/lokeshgile/aasd-audio-gammatones/{int(audio_marker)}.npz"
    cache_path = cache_path_2 if os.path.exists(cache_path_2) else cache_path_1
    
    if os.path.exists(cache_path):
        audio_cache = np.load(cache_path)
        env_l_1d = audio_cache['env_l']
        env_r_1d = audio_cache['env_r']
    else:
        print(f"[INFO] Cache not found. Extracting Gammatones dynamically...")
        from data.extract_gammatone_envelopes import extract_true_gammatone_envelopes
        audio_file = f"/kaggle/input/datasets/lokeshgile/aasd-audio/Stimuli Audio/mixed_{int(audio_marker):03d}.wav"
        if not os.path.exists(audio_file):
            print(f"[FAIL] Audio file {audio_file} not found.")
            sys.exit(1)
        env_l_28, env_r_28 = extract_true_gammatone_envelopes(audio_file, target_fs=64)
        env_l_1d = norm_env(env_l_28).mean(axis=0)
        env_r_1d = norm_env(env_r_28).mean(axis=0)
    
    win_len = int(2.0 * 64)
    stride = int(1.0 * 64)
    
    print("\n| Time Range | GT State | L-Corr | R-Corr | Margin | Pred | Crosses Switch? |")
    print("|------------|----------|--------|--------|--------|------|-----------------|")
    
    for start in range(0, eeg_norm.shape[1] - win_len, stride):
        end = start + win_len
        t_center = (start + win_len/2.0) / 64.0
        t_start = start / 64.0
        t_end = end / 64.0
        
        # Calculate GT State at t_center
        state = current_state
        for st, s_type in zip(switch_times_sec, types_B):
            if t_center >= st:
                state = s_type
                
        # Did it cross a switch?
        crosses = "No"
        for st in switch_times_sec:
            if t_start <= st <= t_end:
                crosses = f"Yes ({st:.1f}s)"
                
        # EEG Window
        w_eeg = eeg_norm[:, start:end]
        w_eeg = (w_eeg - w_eeg.mean(axis=1, keepdims=True)) / (w_eeg.std(axis=1, keepdims=True) + 1e-8)
        
        # Audio Windows
        if end > len(env_l_1d): break
        w_l = env_l_1d[start:end]
        w_r = env_r_1d[start:end]
        w_l = (w_l - w_l.mean()) / (w_l.std() + 1e-8)
        w_r = (w_r - w_r.mean()) / (w_r.std() + 1e-8)
        
        # Inference
        batch_eeg = torch.tensor(w_eeg[np.newaxis, :, :], dtype=torch.float32).to(device)
        with torch.no_grad():
            out = model(batch_eeg)
            pred_env = out.cpu().numpy()[0]
            
        corr_l = safe_corr_np(pred_env[np.newaxis, :], w_l[np.newaxis, :])[0]
        corr_r = safe_corr_np(pred_env[np.newaxis, :], w_r[np.newaxis, :])[0]
        margin = corr_l - corr_r
        pred_state = "L" if margin > 0 else "R"
        
        # Formatting
        gt_fmt = "Left" if state == 'L' else "Right"
        pred_fmt = "Left" if pred_state == "L" else "Right"
        
        # Highlight incorrect predictions (ignoring switch windows for clarity, or just mark them)
        if pred_fmt != gt_fmt and "No" in crosses:
            pred_fmt = f"**{pred_fmt}**" # Mark wrong predictions
            
        print(f"| {t_start:>4.1f}-{t_end:<4.1f}s | {gt_fmt:<8} | {corr_l:>6.3f} | {corr_r:>6.3f} | {margin:>+6.3f} | {pred_fmt:<14} | {crosses:<15} |")

if __name__ == "__main__":
    main()
