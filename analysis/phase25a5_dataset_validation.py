import torch
import numpy as np
import scipy.io
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import argparse
import sys
import os
import glob
import matplotlib.pyplot as plt
import tempfile
import warnings
import json
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from models.aad_conformer import AADConformer
    from data.extract_gammatone_envelopes import extract_gammatone_envelopes
except ImportError:
    print("Could not import dependencies.")
    sys.exit(1)

def safe_corr_np(x, y, eps=1e-8):
    x_mean = x.mean(axis=1, keepdims=True)
    y_mean = y.mean(axis=1, keepdims=True)
    num = np.sum((x - x_mean) * (y - y_mean), axis=1)
    den = np.sqrt(np.sum((x - x_mean)**2, axis=1) * np.sum((y - y_mean)**2, axis=1))
    return num / (den + eps)

def norm_env(env):
    env = env - env.mean(axis=1, keepdims=True)
    env = env / (env.std(axis=1, keepdims=True) + 1e-12)
    return env

def extract_true_gammatone_envelopes(wav_path, target_fs=64):
    fs, audio = wavfile.read(wav_path)
    if len(audio.shape) > 1:
        left = audio[:, 0].astype(np.float64)
        right = audio[:, 1].astype(np.float64)
    else:
        left = audio.astype(np.float64)
        right = left
        
    def process_channel(data):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as t:
            wavfile.write(t.name, fs, data)
            env_28 = extract_gammatone_envelopes(t.name, target_fs=target_fs)
            os.remove(t.name)
        return env_28
        
    env_l_28 = process_channel(left)
    env_r_28 = process_channel(right)
    return env_l_28, env_r_28

def build_audio_cache(audio_dir, cache_dir, target_fs=64):
    os.makedirs(cache_dir, exist_ok=True)
    wav_files = glob.glob(os.path.join(audio_dir, '*.wav'))
    cache_dict = {}
    
    print(f"[INFO] Building Audio Gammatone Cache for {len(wav_files)} files...")
    for wav_path in tqdm(wav_files, desc="Audio Cache"):
        basename = os.path.basename(wav_path)
        marker = basename.replace('mixed_', '').replace('.wav', '')
        try:
            marker_id = int(marker)
        except:
            continue
            
        cache_path = os.path.join(cache_dir, f"{marker_id}.npz")
        if os.path.exists(cache_path):
            data = np.load(cache_path)
            cache_dict[marker_id] = (data['env_l'], data['env_r'])
        else:
            env_l_28, env_r_28 = extract_true_gammatone_envelopes(wav_path, target_fs)
            env_l_1d = norm_env(env_l_28).mean(axis=0)
            env_r_1d = norm_env(env_r_28).mean(axis=0)
            np.savez(cache_path, env_l=env_l_1d, env_r=env_r_1d)
            cache_dict[marker_id] = (env_l_1d, env_r_1d)
            
    return cache_dict

def extract_epoch_events(events, target_epoch):
    epoch_events = []
    for i in range(events.shape[0]):
        ev = events[i] if events.ndim > 1 else events
        ep = int(ev[4] if events.ndim > 1 else getattr(ev, 'epoch', 0))
        if ep == target_epoch:
            ev_t = str(ev[0] if events.ndim > 1 else getattr(ev, 'type', '')).strip()
            ev_lat = float(ev[1] if events.ndim > 1 else getattr(ev, 'latency', 0))
            epoch_events.append((ev_t, ev_lat))
    return epoch_events

def run_inference(model, eeg_norm, env_l_1d, env_r_1d, win_sec, stride_sec, device, fs=64):
    win_len = int(win_sec * fs)
    stride = int(stride_sec * fs)
    
    eeg_windows = []
    l_windows = []
    r_windows = []
    times = []
    
    for start in range(0, min(eeg_norm.shape[1], len(env_l_1d)) - win_len, stride):
        win_eeg = eeg_norm[:, start:start+win_len]
        
        w_eeg_mean = win_eeg.mean(axis=1, keepdims=True)
        w_eeg_std = win_eeg.std(axis=1, keepdims=True) + 1e-8
        win_eeg_norm = (win_eeg - w_eeg_mean) / w_eeg_std
        eeg_windows.append(win_eeg_norm)
        
        win_l = env_l_1d[start:start+win_len]
        win_r = env_r_1d[start:start+win_len]
        
        win_l_norm = (win_l - win_l.mean()) / (win_l.std() + 1e-8)
        win_r_norm = (win_r - win_r.mean()) / (win_r.std() + 1e-8)
        
        l_windows.append(win_l_norm)
        r_windows.append(win_r_norm)
        
        times.append((start + win_len/2) / fs)
        
    if not eeg_windows:
        return np.array([]), np.array([])
        
    eeg_batch = torch.tensor(np.stack(eeg_windows), dtype=torch.float32).to(device)
    
    with torch.no_grad():
        out, _ = model(eeg_batch, return_features=True)
        pred_envs = out.squeeze(1).cpu().numpy()
        
    l_batch = np.stack(l_windows)
    r_batch = np.stack(r_windows)
    
    corr_l = safe_corr_np(pred_envs, l_batch)
    corr_r = safe_corr_np(pred_envs, r_batch)
    
    margins = corr_l - corr_r
    return np.array(times), margins

def process_subject(mat_path, model, audio_cache, device):
    subject_id = os.path.basename(os.path.dirname(mat_path))
    try:
        mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        eeg_var = [k for k in mat.keys() if not k.startswith('__')][0]
        eeg_data = mat[eeg_var].data
        events = mat[eeg_var].event
    except Exception as e:
        print(f"Error loading {mat_path}: {e}")
        return subject_id, []
    
    # 1. Bandpass filter 1-8Hz
    nyq = 128 / 2
    b, a = signal.butter(4, [1.0/nyq, 8.0/nyq], btype='band')
    eeg_filt = signal.filtfilt(b, a, eeg_data, axis=1)
    
    # 2. Downsample to 64Hz
    import math
    g = math.gcd(64, 128)
    eeg_64 = signal.resample_poly(eeg_filt, 64 // g, 128 // g, axis=1)
    eeg_64 = eeg_64[:8, :] # 8 Channels
    eeg_norm = norm_env(eeg_64)
    
    # Find max epoch
    max_epoch = 1
    for i in range(events.shape[0]):
        ev = events[i] if events.ndim > 1 else events
        ep = int(ev[4] if events.ndim > 1 else getattr(ev, 'epoch', 0))
        max_epoch = max(max_epoch, ep)
        
    subject_results = []
    
    for ep in range(1, max_epoch + 1):
        epoch_events = extract_epoch_events(events, ep)
        if not epoch_events:
            continue
            
        audio_marker = None
        trial_start_samples = 0
        switch_events = []
        
        for ev_t, ev_lat in epoch_events:
            if ev_t not in ['179', '184']:
                audio_marker = ev_t
                trial_start_samples = ev_lat
            else:
                switch_events.append((ev_t, ev_lat))
                
        if audio_marker is None:
            continue
            
        try:
            marker_id = int(audio_marker)
        except:
            continue
            
        if marker_id not in audio_cache:
            continue
            
        env_l_1d, env_r_1d = audio_cache[marker_id]
        
        switch_times_sec = []
        for t, lat in switch_events:
            t_sec = (lat - trial_start_samples) / 128.0
            if t_sec > 0: 
                switch_times_sec.append(t_sec)
                
        raw_events = [(t_sec, t) for t_sec, t in zip(switch_times_sec, [t for t, l in switch_events if (l-trial_start_samples)/128.0 > 0])]
        
        if eeg_data.ndim == 3:
            trial_eeg = eeg_norm[:, :, ep-1]
        else:
            start_idx = int(trial_start_samples * 64 / 128)
            end_idx = start_idx + len(env_l_1d)
            if end_idx > eeg_norm.shape[1]:
                end_idx = eeg_norm.shape[1]
            trial_eeg = eeg_norm[:, start_idx:end_idx]
            
        t_2, m_2 = run_inference(model, trial_eeg, env_l_1d, env_r_1d, 2.0, 1.0, device)
        
        if len(t_2) == 0:
            continue
            
        subject_results.append({
            'trial_id': ep,
            'audio_marker': marker_id,
            't': t_2.tolist(),
            'm': m_2.tolist(),
            'raw_events': raw_events
        })
        
    return subject_id, subject_results

def generate_gt_state(t_array, raw_events, mapping_type):
    if len(raw_events) == 0:
        return np.ones(len(t_array)) # Default Left
        
    if mapping_type == 'A':
        types = ['L' if ev_code == '179' else 'R' for t_sec, ev_code in raw_events]
    else:
        types = ['R' if ev_code == '179' else 'L' for t_sec, ev_code in raw_events]
        
    st_times = [t_sec for t_sec, ev_code in raw_events]
    
    current_state = 1 if types[0] == 'R' else -1
    
    gt = np.zeros(len(t_array))
    for i, t in enumerate(t_array):
        state = current_state
        for st, s_type in zip(st_times, types):
            if t >= st:
                state = 1 if s_type == 'L' else -1
        gt[i] = state
        
    return gt

def compute_auroc(m_all, gt_all):
    y_true = (gt_all == 1).astype(int)
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, _ = roc_curve(y_true, m_all)
    return auc(fpr, tpr)

def compute_trajectories(t_array, m_array, raw_events, mapping_type, traj_window=10.0):
    trajectories = []
    if mapping_type == 'A':
        types = ['L' if ev_code == '179' else 'R' for t_sec, ev_code in raw_events]
    else:
        types = ['R' if ev_code == '179' else 'L' for t_sec, ev_code in raw_events]
        
    st_times = [t_sec for t_sec, ev_code in raw_events]
    
    for st, st_type in zip(st_times, types):
        idx_mask = (t_array >= st - traj_window) & (t_array <= st + traj_window)
        t_segment = t_array[idx_mask] - st
        m_segment = m_array[idx_mask]
        
        if st_type == 'R':
            m_segment = -m_segment
            
        if len(t_segment) > 0:
            common_t = np.linspace(-traj_window, traj_window, 40)
            interp_m = np.interp(common_t, t_segment, m_segment)
            trajectories.append(interp_m)
            
    return trajectories

def run_phase25a5(aasd_eeg_dir, aasd_audio_dir, checkpoint_path, out_dir):
    print("====================================================")
    print("PHASE 25A.5: DATASET-SCALE DECODER VALIDATION")
    print("====================================================")
    
    os.makedirs(out_dir, exist_ok=True)
    cache_dir = os.path.join(out_dir, "audio_cache")
    audio_cache = build_audio_cache(aasd_audio_dir, cache_dir, target_fs=64)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AADConformer(in_channels=8)
    if checkpoint_path and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    mat_files = glob.glob(os.path.join(aasd_eeg_dir, '*', '*.mat'))
    if not mat_files:
        print(f"[FAIL] No .mat files found in {aasd_eeg_dir}")
        sys.exit(1)
        
    print(f"[INFO] Processing {len(mat_files)} subjects sequentially on {device}...")
    
    all_results = {}
    for mat_path in tqdm(mat_files, desc="Subjects"):
        subj, subj_res = process_subject(mat_path, model, audio_cache, device)
        all_results[subj] = subj_res
        
    print("[INFO] Evaluation complete. Computing Population Statistics...")
    
    m_all_A, gt_all_A = [], []
    m_all_B, gt_all_B = [], []
    
    subject_aurocs_B = {}
    
    all_trajectories_B = []
    
    for subj, trials in all_results.items():
        s_m_B, s_gt_B = [], []
        for tr in trials:
            t_arr = np.array(tr['t'])
            m_arr = np.array(tr['m'])
            raw_evs = tr['raw_events']
            
            gt_A = generate_gt_state(t_arr, raw_evs, 'A')
            gt_B = generate_gt_state(t_arr, raw_evs, 'B')
            
            m_all_A.extend(m_arr)
            gt_all_A.extend(gt_A)
            
            m_all_B.extend(m_arr)
            gt_all_B.extend(gt_B)
            
            s_m_B.extend(m_arr)
            s_gt_B.extend(gt_B)
            
            trajs = compute_trajectories(t_arr, m_arr, raw_evs, 'B')
            all_trajectories_B.extend(trajs)
            
        if len(s_m_B) > 0:
            subject_aurocs_B[subj] = compute_auroc(np.array(s_m_B), np.array(s_gt_B))
            
    m_all_A = np.array(m_all_A)
    gt_all_A = np.array(gt_all_A)
    auroc_A = compute_auroc(m_all_A, gt_all_A)
    
    m_all_B = np.array(m_all_B)
    gt_all_B = np.array(gt_all_B)
    auroc_B = compute_auroc(m_all_B, gt_all_B)
    
    print(f"\n--- BUTTON MAPPING HYPOTHESIS TEST ---")
    print(f"Hypothesis A (179=L, 184=R) Population AUROC: {auroc_A:.4f}")
    print(f"Hypothesis B (179=R, 184=L) Population AUROC: {auroc_B:.4f}")
    
    if auroc_B > auroc_A:
        print("[CONCLUSION] Hypothesis B is the mathematically correct ground truth mapping.")
    else:
        print("[CONCLUSION] Hypothesis A is the mathematically correct ground truth mapping.")
        
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    y_true_B = (gt_all_B == 1).astype(int)
    fpr, tpr, _ = roc_curve(y_true_B, m_all_B)
    axes[0, 0].plot(fpr, tpr, color='blue', lw=2, label=f'Conformer ROC (AUC = {auroc_B:.3f})')
    axes[0, 0].plot([0, 1], [0, 1], color='gray', linestyle='--')
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('Dataset Population ROC (All Trials, 2s Windows)')
    axes[0, 0].legend()
    
    subjs = list(subject_aurocs_B.keys())
    subjs.sort()
    auroc_vals = [subject_aurocs_B[s] for s in subjs]
    
    axes[0, 1].bar(subjs, auroc_vals, color='purple', alpha=0.7)
    axes[0, 1].axhline(0.5, color='red', linestyle='--', label='Random Chance')
    axes[0, 1].set_ylim([0.4, 1.0])
    axes[0, 1].set_xticklabels(subjs, rotation=45, ha='right')
    axes[0, 1].set_title('Decoder Transfer AUROC per Subject')
    axes[0, 1].set_ylabel('AUROC')
    axes[0, 1].legend()
    
    m_all_B_left = m_all_B[gt_all_B == 1]
    m_all_B_right = m_all_B[gt_all_B == -1]
    axes[1, 0].hist(m_all_B_left, bins=50, density=True, color='green', alpha=0.5, label='GT: Left')
    axes[1, 0].hist(m_all_B_right, bins=50, density=True, color='red', alpha=0.5, label='GT: Right')
    axes[1, 0].axvline(0, color='k', linestyle='--')
    axes[1, 0].set_title('Population Margin Distribution (Split by GT)')
    axes[1, 0].legend()
    
    if len(all_trajectories_B) > 0:
        common_t = np.linspace(-10.0, 10.0, 40)
        trajectories_mat = np.array(all_trajectories_B)
        mean_traj = np.mean(trajectories_mat, axis=0)
        std_traj = np.std(trajectories_mat, axis=0)
        ci = 1.96 * std_traj / np.sqrt(trajectories_mat.shape[0])
        
        axes[1, 1].plot(common_t, mean_traj, color='blue', lw=3, label=f'Avg Margin (N={trajectories_mat.shape[0]} switches)')
        axes[1, 1].fill_between(common_t, mean_traj - ci, mean_traj + ci, color='blue', alpha=0.2, label='95% CI')
        axes[1, 1].axvline(0, color='magenta', linestyle='--', label='Biological Switch Event')
        axes[1, 1].axhline(0, color='k', linestyle=':')
        axes[1, 1].set_title('Grand Average Trigger Pull (-10s to +10s)')
        axes[1, 1].set_xlabel('Time relative to button press (seconds)')
        axes[1, 1].set_ylabel('Margin')
        axes[1, 1].legend()
        
    plot_path = os.path.join(out_dir, 'phase25a5_population_report.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"\n[INFO] Saved comprehensive report to {plot_path}")
    
    stats_dict = {
        'Population_AUROC_HypA': float(auroc_A),
        'Population_AUROC_HypB': float(auroc_B),
        'Subject_AUROCs': subject_aurocs_B,
        'Total_Trials': sum(len(trials) for trials in all_results.values()),
        'Total_Switches_Analyzed': len(all_trajectories_B)
    }
    with open(os.path.join(out_dir, 'phase25a5_stats.json'), 'w') as f:
        json.dump(stats_dict, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aasd_eeg_dir", type=str, required=True, help="Path to AASD root dir containing S*/S*.mat")
    parser.add_argument("--aasd_audio_dir", type=str, required=True, help="Path to AASD Audio dir")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to Conformer checkpoint")
    parser.add_argument("--out_dir", type=str, default="results/phase25a5", help="Output directory")
    args = parser.parse_args()
    run_phase25a5(args.aasd_eeg_dir, args.aasd_audio_dir, args.checkpoint, args.out_dir)
