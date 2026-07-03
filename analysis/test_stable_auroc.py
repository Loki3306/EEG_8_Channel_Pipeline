import numpy as np
import os
import torch
import glob
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

import sys
# Make it work regardless of where it is run from by adding the parent directory to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
from models.aad_conformer import AADConformer

def safe_corr_np(x, y, eps=1e-8):
    x_mean = x.mean(axis=1, keepdims=True)
    y_mean = y.mean(axis=1, keepdims=True)
    num = np.sum((x - x_mean) * (y - y_mean), axis=1)
    den = np.sqrt(np.sum((x - x_mean)**2, axis=1) * np.sum((y - y_mean)**2, axis=1))
    return num / (den + eps)

def compute_auroc(margins, gt):
    if len(margins) == 0:
        return 0.50
    y_true = (gt == 1).astype(int)
    if len(np.unique(y_true)) < 2:
        return 0.50
    try:
        return roc_auc_score(y_true, margins)
    except ValueError:
        return 0.50

def generate_gt_state(t_array, events, hypothesis='A'):
    gt = np.zeros(len(t_array))
    if len(events) == 0:
        return gt
    st_times = []
    types = []
    for ev_t, ev_lat in events:
        if ev_t in ['179', '184']:
            st_times.append(ev_lat / 128.0)
            if hypothesis == 'A':
                types.append('L' if ev_t == '179' else 'R')
            else:
                types.append('R' if ev_t == '179' else 'L')
    if len(types) == 0:
        return gt
    current_state = 1 if types[0] == 'R' else -1
    for i, t in enumerate(t_array):
        state = current_state
        for st, s_type in zip(st_times, types):
            if t >= st:
                state = 1 if s_type == 'L' else -1
        gt[i] = state
    return gt

def get_stable_mask(t_array, events, delay_sec=4.0):
    """
    Returns a boolean mask where True means the window is 'stable'
    (i.e., at least delay_sec AFTER any switch).
    """
    mask = np.ones(len(t_array), dtype=bool)
    switch_times = [ev_lat / 128.0 for ev_t, ev_lat in events if ev_t in ['179', '184']]
    
    # Also mask the very beginning of the trial (first 4 seconds)
    switch_times = [0.0] + switch_times
    
    for st in switch_times:
        for i, t in enumerate(t_array):
            if st <= t < st + delay_sec:
                mask[i] = False
    return mask

def norm_env(env):
    env = env - env.mean(axis=1, keepdims=True)
    env = env / (env.std(axis=1, keepdims=True) + 1e-12)
    return env

def main():
    import scipy.io
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AADConformer(in_channels=8).to(device)
    ckpt = torch.load('/kaggle/input/datasets/lowkieee/eeg-aad-conformer-seed1-checkpoints/checkpoints/seed_1/model_S1.pt', map_location=device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()

    eeg_dir = '/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG'
    audio_dir = '/kaggle/input/datasets/lokeshgile/aasd-audio-gammatones'

    # Neuroscan 64-channel map
    target_channels = ['T7', 'C2', 'FT8', 'P7', 'CPz', 'Fp1', 'TP8', 'C3']
    fallback_map = {'T7': 23, 'C2': 28, 'FT8': 22, 'P7': 41, 'CPz': 36, 'Fp1': 0, 'TP8': 40, 'C3': 25}
    sel_idx = [fallback_map[tc] for tc in target_channels]

    b, a = signal.butter(4, [1.0/32.0, 8.0/32.0], btype='band')
    
    all_margins = []
    all_gt = []
    all_margins_stable = []
    all_gt_stable = []

    # Test just a few subjects to quickly verify
    mat_files = glob.glob(os.path.join(eeg_dir, '*', '*.mat'))[:2]
    
    for mf in mat_files:
        mat = scipy.io.loadmat(mf, squeeze_me=True, struct_as_record=False)
        eeg_var = [k for k in mat.keys() if not k.startswith('__')][0]
        eeg_obj = mat[eeg_var]
        data_all = eeg_obj.data
        events = eeg_obj.event

        def get_ev_attr(e, attr_name, array_idx=0):
            try:
                if hasattr(e, attr_name):
                    return getattr(e, attr_name)
                if isinstance(e, np.ndarray):
                    if e.size == 1 and hasattr(e.flat[0], attr_name):
                        return getattr(e.flat[0], attr_name)
                    return e[array_idx]
            except:
                pass
            return ''

        trial_starts = []
        for i, ev in enumerate(events):
            t_str = str(get_ev_attr(ev, 'type', 0)).strip()
            if t_str and t_str not in ['179', '184', '254', '255']:
                try:
                    lat = float(get_ev_attr(ev, 'latency', 1))
                    trial_starts.append((i, t_str, lat))
                except:
                    pass

        for idx_ev, (ev_idx, audio_marker, trial_start_lat) in enumerate(trial_starts):
            npz_path = os.path.join(audio_dir, f"{int(audio_marker)}.npz")
            if not os.path.exists(npz_path):
                continue
                
            audio_data = np.load(npz_path)
            env_l_1d = audio_data['env_l']
            env_r_1d = audio_data['env_r']

            # Get trial events
            next_start_lat = trial_starts[idx_ev+1][2] if idx_ev+1 < len(trial_starts) else data_all.shape[1]
            raw_evs = []
            for ev in events[ev_idx:]:
                try:
                    lat = float(get_ev_attr(ev, 'latency', 1))
                    if lat >= next_start_lat:
                        break
                    t_str = str(get_ev_attr(ev, 'type', 0)).strip()
                    raw_evs.append((t_str, lat - trial_start_lat))
                except:
                    pass

            trial_data = data_all[:, int(trial_start_lat)-1:int(next_start_lat)-1]
            if len(trial_data.shape) == 3:
                trial_data = trial_data[:, :, 0]
                
            eeg_filt = signal.filtfilt(b, a, trial_data, axis=1)
            eeg_64 = signal.resample_poly(eeg_filt, 1, 2, axis=1)
            eeg_8 = eeg_64[sel_idx, :]

            win_len = 128
            hop = 64
            t_array = np.arange(0, min(eeg_8.shape[1], len(env_l_1d)) - win_len, hop) / 64.0 + 1.0
            
            margins = []
            for start in range(0, min(eeg_8.shape[1], len(env_l_1d)) - win_len, hop):
                win_eeg = eeg_8[:, start:start+win_len]
                win_eeg = (win_eeg - win_eeg.mean(axis=1, keepdims=True)) / (win_eeg.std(axis=1, keepdims=True) + 1e-8)
                
                win_l = env_l_1d[start:start+win_len]
                win_r = env_r_1d[start:start+win_len]
                win_l = (win_l - win_l.mean()) / (win_l.std() + 1e-8)
                win_r = (win_r - win_r.mean()) / (win_r.std() + 1e-8)
                
                eeg_t = torch.tensor(win_eeg[np.newaxis, ...], dtype=torch.float32).to(device)
                with torch.no_grad():
                    out, _ = model(eeg_t, return_features=True)
                    pred_env = out.squeeze(1).cpu().numpy()
                    
                c_l = safe_corr_np(pred_env, win_l[np.newaxis, ...])[0]
                c_r = safe_corr_np(pred_env, win_r[np.newaxis, ...])[0]
                margins.append(c_l - c_r)

            gt_B = generate_gt_state(t_array, raw_evs, 'B')
            stable_mask = get_stable_mask(t_array, raw_evs, delay_sec=4.0)

            all_margins.extend(margins)
            all_gt.extend(gt_B)
            
            all_margins_stable.extend(np.array(margins)[stable_mask])
            all_gt_stable.extend(gt_B[stable_mask])

    print(f"All AUROC: {compute_auroc(all_margins, all_gt):.4f}")
    print(f"Stable AUROC: {compute_auroc(all_margins_stable, all_gt_stable):.4f}")

if __name__ == "__main__":
    main()
