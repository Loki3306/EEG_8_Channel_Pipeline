import torch
import numpy as np
import scipy.io
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import argparse
import sys
import os
import matplotlib.pyplot as plt
import tempfile
import warnings
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

def norm_env(env):
    env = env - env.mean(axis=1, keepdims=True)
    env = env / (env.std(axis=1, keepdims=True) + 1e-12)
    return env

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
        
        # Center of the window in seconds
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

def run_phase25a(aasd_eeg_path, aasd_audio_path, checkpoint_path, out_dir):
    print("====================================================")
    print("PHASE 25A: GROUND TRUTH BIOLOGICAL ALIGNMENT")
    print("====================================================")
    
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = AADConformer(in_channels=8)
    if checkpoint_path and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    print("[INFO] Loading AASD Trial 1...")
    mat = scipy.io.loadmat(aasd_eeg_path, squeeze_me=True, struct_as_record=False)
    eeg_var = [k for k in mat.keys() if not k.startswith('__')][0]
    eeg_data = mat[eeg_var].data
    events = mat[eeg_var].event
    
    trial_eeg_128 = eeg_data[:, :, 0]
    
    nyq = 128 / 2
    b, a = signal.butter(4, [1.0/nyq, 8.0/nyq], btype='band')
    trial_eeg_128_filt = signal.filtfilt(b, a, trial_eeg_128, axis=1)
    
    import math
    g = math.gcd(64, 128)
    trial_eeg_64 = signal.resample_poly(trial_eeg_128_filt, 64 // g, 128 // g, axis=1)
    trial_eeg_64 = trial_eeg_64[:8, :]
    eeg_norm = norm_env(trial_eeg_64)
    
    # Extract Ground Truth
    audio_marker = None
    trial_start_samples = 0
    switch_events = []
    
    for i in range(events.shape[0]):
        ev = events[i] if events.ndim > 1 else events
        ep = int(ev[4] if events.ndim > 1 else getattr(ev, 'epoch', 0))
        if ep == 1:
            ev_t = str(ev[0] if events.ndim > 1 else getattr(ev, 'type', '')).strip()
            ev_lat = float(ev[1] if events.ndim > 1 else getattr(ev, 'latency', 0))
            if ev_t not in ['179', '184']:
                audio_marker = ev_t
                trial_start_samples = ev_lat
            else:
                switch_events.append((ev_t, ev_lat))
                
    if audio_marker is None:
        print("[FAIL] Audio marker not found.")
        sys.exit(1)
        
    audio_file = os.path.join(aasd_audio_path, f"mixed_{int(audio_marker):03d}.wav")
    if not os.path.exists(audio_file):
        print(f"[FAIL] Audio file {audio_file} not found.")
        sys.exit(1)
        
    print(f"[INFO] Extracting TRUE Gammatone envelopes from {audio_file}...")
    env_l_28, env_r_28 = extract_true_gammatone_envelopes(audio_file, target_fs=64)
    env_l_1d = norm_env(env_l_28).mean(axis=0)
    env_r_1d = norm_env(env_r_28).mean(axis=0)
    
    # Process Switch Events
    switch_times_sec = []
    switch_types = [] # 'L' or 'R'
    for t, lat in switch_events:
        t_sec = (lat - trial_start_samples) / 128.0 # Original EEG was 128Hz
        switch_times_sec.append(t_sec)
        switch_types.append('L' if t == '179' else 'R')
        
    print(f"[INFO] Found {len(switch_times_sec)} switch events: {list(zip(switch_times_sec, switch_types))}")
    
    # Run dual resolution inference
    print("[INFO] Running 10s Window Inference...")
    t_10, m_10 = run_inference(model, eeg_norm, env_l_1d, env_r_1d, 10.0, 1.0, device)
    
    print("[INFO] Running 2s Window Inference...")
    t_2, m_2 = run_inference(model, eeg_norm, env_l_1d, env_r_1d, 2.0, 1.0, device)
    
    # Helper to determine Ground Truth State at time t
    def get_gt_state(t):
        if len(switch_times_sec) == 0:
            return 1 # Default Left
        
        # Assume initial state is opposite of first switch
        current_state = 1 if switch_types[0] == 'R' else -1
        
        for st, s_type in zip(switch_times_sec, switch_types):
            if t >= st:
                current_state = 1 if s_type == 'L' else -1
        return current_state

    gt_10 = np.array([get_gt_state(t) for t in t_10])
    gt_2 = np.array([get_gt_state(t) for t in t_2])
    
    # Compute Average Switch Trajectory (using 2s windows for temporal precision)
    traj_window = 10.0 # seconds before and after
    trajectories = []
    
    for st, st_type in zip(switch_times_sec, switch_types):
        idx_mask = (t_2 >= st - traj_window) & (t_2 <= st + traj_window)
        t_segment = t_2[idx_mask] - st
        m_segment = m_2[idx_mask]
        
        # If the switch is TO RIGHT (st_type=='R'), a successful margin goes from + to -.
        # Let's align them so that successful transition is always going "up" (from - to +)
        if st_type == 'R':
            m_segment = -m_segment
            
        if len(t_segment) > 0:
            # Interpolate to a common time grid
            common_t = np.linspace(-traj_window, traj_window, 40)
            interp_m = np.interp(common_t, t_segment, m_segment)
            trajectories.append(interp_m)
            
    avg_trajectory = np.mean(trajectories, axis=0) if trajectories else None
    
    # Visualization
    fig, axes = plt.subplots(4, 1, figsize=(15, 16))
    
    def plot_timeline(ax, t, m, gt, title):
        ax.plot(t, m, label='Margin (L - R)', color='b')
        ax.axhline(0, color='k', linestyle='--')
        ax.set_title(title)
        ax.set_ylabel('Pearson Margin')
        
        # Shade background
        ax.fill_between(t, -1, 1, where=(gt == 1), color='green', alpha=0.1, label='GT: Attending Left')
        ax.fill_between(t, -1, 1, where=(gt == -1), color='red', alpha=0.1, label='GT: Attending Right')
        ax.set_ylim([min(m)-0.1, max(m)+0.1])
        
        for st, st_type in zip(switch_times_sec, switch_types):
            ax.axvline(st, color='magenta', linestyle=':', lw=2)
            ax.text(st, max(m), f' Switch {st_type}', color='magenta', rotation=90, va='top')
            
        ax.legend(loc='upper right')
        
    plot_timeline(axes[0], t_10, m_10, gt_10, '10s Window Decoder Output (High SNR)')
    plot_timeline(axes[1], t_2, m_2, gt_2, '2s Window Controller Input (High Variance, Low Latency)')
    
    # Histograms split by Ground Truth
    m_10_left = m_10[gt_10 == 1]
    m_10_right = m_10[gt_10 == -1]
    
    if len(m_10_left) > 0:
        axes[2].hist(m_10_left, bins=20, color='green', alpha=0.6, label='GT: Left')
    if len(m_10_right) > 0:
        axes[2].hist(m_10_right, bins=20, color='red', alpha=0.6, label='GT: Right')
    axes[2].axvline(0, color='k', linestyle='--')
    axes[2].set_title('Margin Distribution split by Ground Truth (10s windows)')
    axes[2].legend()
    
    # Average Switch Trajectory
    if avg_trajectory is not None:
        common_t = np.linspace(-traj_window, traj_window, 40)
        axes[3].plot(common_t, avg_trajectory, color='purple', lw=3, label='Avg Normalized Trajectory')
        axes[3].axvline(0, color='magenta', linestyle='--', label='Switch Event')
        axes[3].axhline(0, color='k', linestyle=':')
        axes[3].set_title('Average Trigger Pull: Decoder Response to Biological Switch (Aligned so target is positive)')
        axes[3].set_xlabel('Time relative to button press (seconds)')
        axes[3].set_ylabel('Margin')
        axes[3].legend()
        
    plot_path = os.path.join(out_dir, 'phase25a_decoder_transfer.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"[INFO] Saved plots to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--aasd_eeg", type=str, required=True, help="Path to AASD S18.mat")
    parser.add_argument("--aasd_audio", type=str, required=True, help="Path to AASD Audio dir")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to Conformer checkpoint")
    parser.add_argument("--out_dir", type=str, default="results/phase25", help="Output directory")
    args = parser.parse_args()
    run_phase25a(args.aasd_eeg, args.aasd_audio, args.checkpoint, args.out_dir)
