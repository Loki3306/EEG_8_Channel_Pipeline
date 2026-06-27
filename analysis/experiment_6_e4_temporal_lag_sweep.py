import os
import sys
import argparse
import scipy.io as sio
import scipy.io.wavfile as wavfile
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.signal import resample, butter, filtfilt
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add repository root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.matchnet import ContrastiveMatchNet
except ImportError as e:
    print(f"Could not import MatchNet: {e}")
    ContrastiveMatchNet = None

try:
    from data.extract_gammatone_envelopes import extract_gammatone_envelopes
except ImportError as e:
    print(f"Could not import gammatone extraction: {e}")
    extract_gammatone_envelopes = None

# --- Normalization ---
def normalize_array(arr):
    arr = arr - arr.mean(axis=0, keepdims=True)
    scale = arr.std(axis=0, keepdims=True) + 1e-12
    return arr / scale

def main():
    print("\n" + "="*60)
    print("PHASE KUL-6: TEMPORAL LAG SWEEP FOR DTU -> KUL GENERALIZATION")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Load Original Model - HARDCODED as requested
    chk_path = "/kaggle/working/EEG_8_Channel_Pipeline/checkpoints/matchnet_fold_S2_data_preproc_best.pth"
    if not os.path.exists(chk_path):
        chk_path = "checkpoints/matchnet_fold_S2_data_preproc_best.pth"
        
    if not os.path.exists(chk_path):
        print(f"ERROR: DTU MatchNet checkpoint not found at {chk_path}!")
        return
        
    print(f"Loading DTU Checkpoint: {chk_path}")
    model = ContrastiveMatchNet(eeg_model_type="eegnet", eeg_channels=8, audio_channels=28, latent_dim=64).to(device)
    model.load_state_dict(torch.load(chk_path, map_location=device))
    model.eval()
    
    # 2. Config
    mat_path = "/kaggle/input/datasets/lowk1ee/s1-klu/S1_KLU.mat"
    wav_dir = "/kaggle/input/datasets/lowk1ee/audio-klu"
    target_channels = ['Fp1', 'Fp2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8']
    fs_dtu = 64
    window_sec, stride_sec = 3.0, 1.5
    win_samples = int(window_sec * fs_dtu)
    stride_samples = int(stride_sec * fs_dtu)
    
    # Setup lags to sweep: -750 ms to +750 ms in 31.25 ms steps (2 samples at 64Hz)
    lags_samples = list(range(-48, 49, 2))
    lags_ms = [lag * (1000.0 / fs_dtu) for lag in lags_samples]
    
    min_lag = min(lags_samples)
    max_lag = max(lags_samples)
    
    print("\nLoading KUL MAT file...")
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    trials = mat['trials'] if 'trials' in mat else mat['trial']
    
    print("Extracting trials using IDENTICAL DTU Preprocessing (1-8 Hz filter & Gammatone Envelopes)...")
    audio_cache = {}
    def find_wav(name):
        for r, d, f in os.walk(wav_dir):
            if name in f: return os.path.join(r, name)
            if name+".wav" in f: return os.path.join(r, name+".wav")
        return None

    trial_data = []
    
    for t_idx, trial in enumerate(trials):
        print(f"  [Trial {t_idx+1:02d}/{len(trials):02d}] Extracting...")
        eeg_data = trial.RawData.EegData
        fs_eeg = trial.FileHeader.SampleRate
        channel_names = [ch.Label for ch in trial.FileHeader.Channels]
        
        selected_indices = [channel_names.index(tc) if tc in channel_names else [c.upper() for c in channel_names].index(tc.upper()) for tc in target_channels]
        eeg_8 = eeg_data[:, selected_indices]
        
        # EXACT DTU FILTERING: 1.0 - 8.0 Hz (Not 1.0 - 6.0 Hz)
        nyq = 0.5 * fs_eeg
        b, a = butter(2, [1.0/nyq, 8.0/nyq], btype='band')
        eeg_8 = filtfilt(b, a, eeg_8, axis=0)
        eeg_64 = resample(eeg_8, int(len(eeg_8) * fs_dtu / fs_eeg), axis=0)
        
        att_ear = trial.attended_ear
        stimuli = trial.stimuli
        att_wav_name = stimuli[0] if att_ear == 'L' else stimuli[1]
        unatt_wav_name = stimuli[1] if att_ear == 'L' else stimuli[0]
        
        att_wav_path = find_wav(str(att_wav_name))
        unatt_wav_path = find_wav(str(unatt_wav_name))
        
        if not att_wav_path or not unatt_wav_path:
            print(f"    WARNING: Missing audio for Trial {t_idx}. Skipping.")
            continue
            
        def get_cached_env(wav_path):
            if wav_path in audio_cache:
                return audio_cache[wav_path]
            # EXACT DTU ENVELOPE PIPELINE
            env = extract_gammatone_envelopes(wav_path, num_bands=28, low_freq=50, high_freq=8000, target_fs=fs_dtu)
            audio_cache[wav_path] = env
            return env
            
        env_att = get_cached_env(att_wav_path)
        env_unatt = get_cached_env(unatt_wav_path)
        
        eeg_norm = normalize_array(eeg_64) 
        env_att = normalize_array(env_att.T).T 
        env_unatt = normalize_array(env_unatt.T).T
        
        trial_data.append((eeg_norm, env_att, env_unatt))
        
    print(f"\nSuccessfully loaded {len(trial_data)} trials.")
    
    # Sanity Checks
    avg_eeg_len = np.mean([t[0].shape[0] for t in trial_data])
    avg_env_len = np.mean([t[1].shape[1] for t in trial_data])
    
    valid_windows_per_trial = []
    for eeg, env_a, _ in trial_data:
        eeg_len = len(eeg)
        env_len = env_a.shape[1]
        
        audio_start_min = max(0, -min_lag)
        audio_start_max = min(env_len - win_samples, eeg_len - win_samples - max_lag)
        
        windows = len(list(range(audio_start_min, audio_start_max + 1, stride_samples)))
        valid_windows_per_trial.append(max(0, windows))
        
    total_valid_windows = sum(valid_windows_per_trial)
    
    print("\n" + "="*40)
    print("SANITY CHECKS")
    print(f"Average EEG Length   : {avg_eeg_len:.1f} samples ({avg_eeg_len/fs_dtu:.1f} sec)")
    print(f"Average Audio Length : {avg_env_len:.1f} samples ({avg_env_len/fs_dtu:.1f} sec)")
    print(f"Evaluated Windows    : {total_valid_windows} (Strictly constrained to be identical across ALL lags)")
    print("="*40 + "\n")
    
    if total_valid_windows == 0:
        print("ERROR: Lag boundaries are too wide to extract any valid windows. Reduce lag sweep range.")
        return
        
    results = []
    
    with torch.no_grad():
        for i, lag in enumerate(lags_samples):
            lag_ms = lags_ms[i]
            
            all_sim_a = []
            all_sim_b = []
            
            # Evaluate across all trials
            for eeg, env_a, env_b in trial_data:
                eeg_len = len(eeg)
                env_len = env_a.shape[1]
                
                # To guarantee we evaluate the exact same audio segments for every lag,
                # we constrain the audio start index such that audio_start + lag is valid 
                # for ALL evaluated lags.
                audio_start_min = max(0, -min_lag)
                audio_start_max = min(env_len - win_samples, eeg_len - win_samples - max_lag)
                
                for start_audio in range(audio_start_min, audio_start_max + 1, stride_samples):
                    start_eeg = start_audio + lag
                    
                    x = eeg[start_eeg:start_eeg+win_samples].T 
                    ya = env_a[:, start_audio:start_audio+win_samples] 
                    yb = env_b[:, start_audio:start_audio+win_samples]
                    
                    x_t = torch.FloatTensor(x).unsqueeze(0).to(device)
                    ya_t = torch.FloatTensor(ya).unsqueeze(0).to(device)
                    yb_t = torch.FloatTensor(yb).unsqueeze(0).to(device)
                    
                    z_eeg, z_a, z_b = model(x_t, ya_t, yb_t)
                    
                    sim_a = F.cosine_similarity(z_eeg, z_a, dim=1).mean(dim=1).item()
                    sim_b = F.cosine_similarity(z_eeg, z_b, dim=1).mean(dim=1).item()
                    
                    all_sim_a.append(sim_a)
                    all_sim_b.append(sim_b)
            
            all_sim_a = np.array(all_sim_a)
            all_sim_b = np.array(all_sim_b)
            margins = all_sim_a - all_sim_b
            
            mean_sim_a = all_sim_a.mean()
            mean_sim_b = all_sim_b.mean()
            
            mean_delta = margins.mean()
            std_delta = margins.std()
            median_delta = np.median(margins)
            q25_delta = np.percentile(margins, 25)
            q75_delta = np.percentile(margins, 75)
            
            mean_margin = margins.mean() # Same as delta for 2-class setup, kept for consistency
            std_margin = margins.std()
            median_margin = np.median(margins)
            q25_margin = np.percentile(margins, 25)
            q75_margin = np.percentile(margins, 75)
            
            window_acc = (margins > 0).mean()
            
            print(f"Lag {lag:3d} smp | {lag_ms:6.1f} ms | WinAcc: {window_acc*100:5.1f}% | Marg: {mean_margin:7.4f} | Delta: {mean_delta:7.4f} | SimA: {mean_sim_a:6.4f} | SimB: {mean_sim_b:6.4f}")
            
            results.append({
                'Lag_Samples': lag,
                'Lag_ms': lag_ms,
                'Window_Accuracy': window_acc,
                'Mean_Margin': mean_margin,
                'Std_Margin': std_margin,
                'Median_Margin': median_margin,
                'Q25_Margin': q25_margin,
                'Q75_Margin': q75_margin,
                'Mean_Delta': mean_delta,
                'Std_Delta': std_delta,
                'Median_Delta': median_delta,
                'Q25_Delta': q25_delta,
                'Q75_Delta': q75_delta,
                'Mean_SimA': mean_sim_a,
                'Mean_SimB': mean_sim_b
            })

    # Save to CSV
    os.makedirs("analysis", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv("analysis/lag_sweep_results.csv", index=False)
    print("\nSaved numeric results to analysis/lag_sweep_results.csv")
    
    # Visualizations
    print("Generating Visualization Plots...")
    os.makedirs("analysis/figures/lag_sweep", exist_ok=True)
    
    lags = df['Lag_ms']
    
    # 1. Window Accuracy Plot
    plt.figure(figsize=(10, 6))
    plt.plot(lags, df['Window_Accuracy'] * 100, marker='o', linewidth=2, color='blue')
    plt.axhline(50, color='gray', linestyle='dashed', alpha=0.7)
    plt.axvline(0, color='gray', linestyle='dashed', alpha=0.7)
    plt.title("Temporal Lag vs Zero-Shot Window Accuracy")
    plt.xlabel("Lag (ms) [Positive = EEG lags Audio]")
    plt.ylabel("Window Accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.savefig("analysis/figures/lag_sweep/lag_vs_window_accuracy.png")
    
    # 2. Mean Margin Plot
    plt.figure(figsize=(10, 6))
    plt.plot(lags, df['Mean_Margin'], marker='o', linewidth=2, color='green', label='Mean Margin')
    plt.axhline(0, color='gray', linestyle='dashed', alpha=0.7)
    plt.axvline(0, color='gray', linestyle='dashed', alpha=0.7)
    plt.title("Temporal Lag vs Similarity Margin")
    plt.xlabel("Lag (ms) [Positive = EEG lags Audio]")
    plt.ylabel("Margin")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("analysis/figures/lag_sweep/lag_vs_margin.png")
    
    # 3. Mean Delta with Error Bars Plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(lags, df['Mean_Delta'], yerr=df['Std_Delta'], fmt='-o', color='purple', capsize=5, label='Mean Delta ± Std')
    plt.axhline(0, color='gray', linestyle='dashed', alpha=0.7)
    plt.axvline(0, color='gray', linestyle='dashed', alpha=0.7)
    plt.title("Temporal Lag vs Delta (SimA - SimB) with Uncertainty")
    plt.xlabel("Lag (ms) [Positive = EEG lags Audio]")
    plt.ylabel("Delta (SimA - SimB)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("analysis/figures/lag_sweep/lag_vs_delta_errorbar.png")
    
    # 4. SimA and SimB Plot
    plt.figure(figsize=(10, 6))
    plt.plot(lags, df['Mean_SimA'], marker='^', linewidth=2, color='forestgreen', label='Attended Sim (SimA)')
    plt.plot(lags, df['Mean_SimB'], marker='v', linewidth=2, color='firebrick', label='Unattended Sim (SimB)')
    plt.axvline(0, color='gray', linestyle='dashed', alpha=0.7)
    plt.title("Temporal Lag vs Absolute Cosine Similarities")
    plt.xlabel("Lag (ms) [Positive = EEG lags Audio]")
    plt.ylabel("Cosine Similarity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("analysis/figures/lag_sweep/lag_vs_similarities.png")
    
    print("Saved plots to analysis/figures/lag_sweep/")
    print("\nSweep Complete!")

if __name__ == "__main__":
    main()
