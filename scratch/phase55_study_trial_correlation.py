import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr
from pathlib import Path
import os

# -------------------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------------------
SR = 128
DELAYS = np.arange(0, int(0.250 * SR)) # 0 to 250ms delays
WIN_SEC = 2.0
WIN_SAMPLES = int(WIN_SEC * SR)

def create_delayed_features(eeg, delays):
    """
    Creates time-delayed features for linear mapping.
    eeg: (channels, time)
    returns: (time, channels * len(delays))
    """
    C, T = eeg.shape
    features = np.zeros((T, C * len(delays)))
    for i, d in enumerate(delays):
        if d == 0:
            features[:, i*C:(i+1)*C] = eeg.T
        else:
            features[d:, i*C:(i+1)*C] = eeg[:, :-d].T
    return features

def get_attended_speaker_vector(T, switch_points):
    """
    Returns an array of shape (T,) containing 'L' or 'R' for every time step.
    """
    attended = np.full(T, 'L', dtype=object)
    for i in range(len(switch_points)):
        spk, idx = switch_points[i]
        next_idx = switch_points[i+1][1] if i+1 < len(switch_points) else T
        attended[idx:next_idx] = spk
    return attended

def extract_trial_data(trial):
    eeg = trial['eeg'].numpy()
    env_l = trial['env_l'].numpy()
    env_r = trial['env_r'].numpy()
    switch_points = trial['meta']['switch_points']
    
    # Normalize
    eeg = (eeg - np.mean(eeg, axis=1, keepdims=True)) / (np.std(eeg, axis=1, keepdims=True) + 1e-8)
    env_l = (env_l - np.mean(env_l)) / (np.std(env_l) + 1e-8)
    env_r = (env_r - np.mean(env_r)) / (np.std(env_r) + 1e-8)
    
    T = eeg.shape[1]
    
    # Create target (Attended Envelope)
    attended_spk = get_attended_speaker_vector(T, switch_points)
    target_env = np.zeros(T)
    unatt_env = np.zeros(T)
    for t in range(T):
        if attended_spk[t] == 'L':
            target_env[t] = env_l[t]
            unatt_env[t] = env_r[t]
        else:
            target_env[t] = env_r[t]
            unatt_env[t] = env_l[t]
            
    features = create_delayed_features(eeg, DELAYS)
    
    return features, target_env, unatt_env, attended_spk

def main():
    cache_path = Path('/kaggle/working/eeg_cache/S1_processed.pt')
    if not cache_path.exists():
        print("Cache not found. Please run generate_aasd_cache.py first.")
        return
        
    print("\n=======================================================")
    print(" PHASE 55: SINGLE-TRIAL ENVELOPE CORRELATION STUDY")
    print("=======================================================\n")
    
    print("Loading S1 Cache...")
    cached = torch.load(cache_path, map_location='cpu', weights_only=False)
    trials = cached['raw']
    
    # We will study Trial 0. We train on Trials 1 to N.
    print("Extracting features for training (Trials 1+)...")
    X_train = []
    y_train = []
    
    for i in range(1, len(trials)):
        feat, targ, _, _ = extract_trial_data(trials[i])
        # Drop the first 250ms because delays are zero-padded
        drop_idx = len(DELAYS)
        X_train.append(feat[drop_idx:])
        y_train.append(targ[drop_idx:])
        
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    
    print(f"Training Ridge Regression (Alpha=100) on {X_train.shape[0]} samples...")
    model = Ridge(alpha=100.0)
    model.fit(X_train, y_train)
    
    print("Reconstructing Audio Envelope for Trial 0...")
    feat_0, targ_0, unatt_0, spk_0 = extract_trial_data(trials[0])
    
    pred_0 = model.predict(feat_0)
    
    # Compute overall correlation
    drop_idx = len(DELAYS)
    corr_att, _ = pearsonr(pred_0[drop_idx:], targ_0[drop_idx:])
    corr_unatt, _ = pearsonr(pred_0[drop_idx:], unatt_0[drop_idx:])
    print(f"Overall Trial 0 Correlation (Attended): {corr_att:.4f}")
    print(f"Overall Trial 0 Correlation (Unattended): {corr_unatt:.4f}")
    
    print("\nComputing sliding 2.0s window correlation...")
    T = len(pred_0)
    times = []
    r_att = []
    r_unatt = []
    
    for start in range(drop_idx, T - WIN_SAMPLES, int(SR * 0.5)): # 0.5s stride
        end = start + WIN_SAMPLES
        
        c_att, _ = pearsonr(pred_0[start:end], targ_0[start:end])
        c_unatt, _ = pearsonr(pred_0[start:end], unatt_0[start:end])
        
        times.append(start / SR)
        r_att.append(c_att)
        r_unatt.append(c_unatt)
        
    # Plotting
    os.makedirs('/kaggle/working/plots', exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    plt.plot(times, r_att, label='Corr w/ Attended', color='blue')
    plt.plot(times, r_unatt, label='Corr w/ Unattended', color='red')
    plt.axhline(0, color='black', linestyle='--')
    
    # Mark switches
    switches = [idx / SR for _, idx in trials[0]['meta']['switch_points']]
    for sw in switches:
        plt.axvline(sw, color='green', linestyle=':', alpha=0.5, label='Attention Switch' if sw==switches[0] else "")
        
    plt.title('Trial 0: Sliding 2.0s Envelope Correlation')
    plt.xlabel('Time (s)')
    plt.ylabel('Pearson r')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/kaggle/working/plots/trial0_correlation.png', dpi=300)
    
    # Plot a tiny 5-second slice of the actual waveforms
    slice_start = 10 * SR
    slice_end = 15 * SR
    
    time_axis = np.arange(slice_start, slice_end) / SR
    
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, targ_0[slice_start:slice_end], label='True Attended', alpha=0.5, color='blue')
    plt.plot(time_axis, pred_0[slice_start:slice_end], label='Reconstructed', alpha=0.8, color='orange')
    plt.title('Trial 0: Reconstructed vs True Envelope (10s to 15s)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (Z-score)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/kaggle/working/plots/trial0_waveform.png', dpi=300)
    
    print("\nVisualizations saved to:")
    print("  /kaggle/working/plots/trial0_correlation.png")
    print("  /kaggle/working/plots/trial0_waveform.png")

if __name__ == "__main__":
    main()
