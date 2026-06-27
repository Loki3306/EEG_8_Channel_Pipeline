import os
import sys
import argparse
import scipy.io as sio
import scipy.io.wavfile as wavfile
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.signal import resample, butter, filtfilt
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

# Add repository root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.matchnet import ContrastiveMatchNet, contrastive_loss
except ImportError as e:
    print(f"Could not import MatchNet: {e}")
    ContrastiveMatchNet = None
    
# --- ERB Filterbank Functions ---
def erb_space(low_freq, high_freq, num_bands):
    erb_low = 21.4 * np.log10(4.37 * low_freq / 1000 + 1)
    erb_high = 21.4 * np.log10(4.37 * high_freq / 1000 + 1)
    erb_points = np.linspace(erb_low, erb_high, num_bands)
    cf = (10 ** (erb_points / 21.4) - 1) * 1000 / 4.37
    return cf

def get_erb_bands(cfs):
    bws = 24.7 * (4.37 * cfs / 1000 + 1)
    lows = cfs - bws / 2
    highs = cfs + bws / 2
    return lows, highs

def apply_bandpass(data, low, high, fs, order=2):
    nyq = 0.5 * fs
    low = max(low / nyq, 0.001)
    high = min(high / nyq, 0.999)
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def _process_band(audio_data, low, high, fs_in):
    band_audio = apply_bandpass(audio_data, low, high, fs_in)
    return np.abs(band_audio) ** 0.3

def extract_28_band_envelope(audio_data, fs_in, fs_out=64, num_bands=28):
    cfs = erb_space(50, 8000, num_bands)
    lows, highs = get_erb_bands(cfs)
    
    from joblib import Parallel, delayed
    envelopes = Parallel(n_jobs=-1)(
        delayed(_process_band)(audio_data, lows[i], highs[i], fs_in) 
        for i in range(num_bands)
    )
    envelopes = np.array(envelopes)
    
    import math
    from scipy.signal import resample_poly
    g = math.gcd(fs_out, fs_in)
    up = fs_out // g
    down = fs_in // g
    
    return resample_poly(envelopes, up, down, axis=1)

# --- Normalization ---
def normalize_array(arr):
    arr = arr - arr.mean(axis=0, keepdims=True)
    scale = arr.std(axis=0, keepdims=True) + 1e-12
    return arr / scale

def find_checkpoint(root_dir):
    candidates = []
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.endswith('.pth') or f.endswith('.pt'):
                # We want the ORIGINAL zero-shot DTU model, not the finetuned ones
                if 'finetuned' not in f.lower():
                    candidates.append(os.path.join(root, f))
    for c in candidates:
        if 'best' in c.lower() or 'matchnet' in c.lower(): return c
    return candidates[0] if candidates else None

def main():
    print("\n" + "="*60)
    print("PHASE KUL-6: TEMPORAL LAG SWEEP FOR DTU -> KUL GENERALIZATION")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Load Original Model
    checkpoint_dir = "/kaggle/working/EEG_8_Channel_Pipeline/checkpoints"
    if not os.path.exists(checkpoint_dir):
        checkpoint_dir = "checkpoints" 
        
    chk_path = find_checkpoint(checkpoint_dir)
    if not chk_path:
        chk_path = find_checkpoint(".")
        
    if not chk_path:
        print("ERROR: No DTU MatchNet checkpoint found!")
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
    
    # Setup lags to sweep: -32 to +32 samples in steps of 4
    # Positive lag: EEG window is shifted FORWARD in time relative to Audio window (EEG lags audio).
    lags_samples = list(range(-32, 33, 4))
    lags_ms = [lag * (1000.0 / fs_dtu) for lag in lags_samples]
    
    print("\nLoading KUL MAT file...")
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    trials = mat['trials'] if 'trials' in mat else mat['trial']
    
    print("Extracting trials and computing envelopes...")
    audio_cache = {}
    def find_wav(name):
        for r, d, f in os.walk(wav_dir):
            if name in f: return os.path.join(r, name)
            if name+".wav" in f: return os.path.join(r, name+".wav")
        return None

    # Pre-extract all trials to memory for fast sweep
    trial_data = []
    
    for t_idx, trial in enumerate(trials):
        print(f"  [Trial {t_idx+1:02d}/{len(trials):02d}] Extracting...")
        eeg_data = trial.RawData.EegData
        fs_eeg = trial.FileHeader.SampleRate
        channel_names = [ch.Label for ch in trial.FileHeader.Channels]
        
        selected_indices = [channel_names.index(tc) if tc in channel_names else [c.upper() for c in channel_names].index(tc.upper()) for tc in target_channels]
        eeg_8 = eeg_data[:, selected_indices]
        
        nyq = 0.5 * fs_eeg
        b, a = butter(2, [1.0/nyq, 6.0/nyq], btype='band')
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
            fs, audio = wavfile.read(wav_path)
            if len(audio.shape) > 1: audio = audio.mean(axis=1)
            env = extract_28_band_envelope(audio, fs, fs_out=64, num_bands=28)
            audio_cache[wav_path] = env
            return env
            
        env_att = get_cached_env(att_wav_path)
        env_unatt = get_cached_env(unatt_wav_path)
        
        eeg_norm = normalize_array(eeg_64) 
        env_att = normalize_array(env_att.T).T 
        env_unatt = normalize_array(env_unatt.T).T
        
        trial_data.append((eeg_norm, env_att, env_unatt))
        
    print(f"\nSuccessfully loaded {len(trial_data)} trials. Starting lag sweep...")
    
    results = []
    
    with torch.no_grad():
        for i, lag in enumerate(lags_samples):
            lag_ms = lags_ms[i]
            
            all_sim_a = []
            all_sim_b = []
            correct_trials = 0
            total_trials = 0
            
            # For this lag, we evaluate across all trials
            for eeg, env_a, env_b in trial_data:
                eeg_len = len(eeg)
                env_len = env_a.shape[1]
                
                trial_sim_a = []
                trial_sim_b = []
                
                # Audio is our anchor index
                for start_audio in range(0, env_len - win_samples + 1, stride_samples):
                    start_eeg = start_audio + lag
                    
                    # Ensure both windows are within bounds
                    if start_eeg < 0 or start_eeg + win_samples > eeg_len:
                        continue
                        
                    x = eeg[start_eeg:start_eeg+win_samples].T 
                    ya = env_a[:, start_audio:start_audio+win_samples] 
                    yb = env_b[:, start_audio:start_audio+win_samples]
                    
                    x_t = torch.FloatTensor(x).unsqueeze(0).to(device)
                    ya_t = torch.FloatTensor(ya).unsqueeze(0).to(device)
                    yb_t = torch.FloatTensor(yb).unsqueeze(0).to(device)
                    
                    z_eeg, z_a, z_b = model(x_t, ya_t, yb_t)
                    
                    sim_a = F.cosine_similarity(z_eeg, z_a, dim=1).mean(dim=1).item()
                    sim_b = F.cosine_similarity(z_eeg, z_b, dim=1).mean(dim=1).item()
                    
                    trial_sim_a.append(sim_a)
                    trial_sim_b.append(sim_b)
                
                if len(trial_sim_a) > 0:
                    trial_sim_a = np.array(trial_sim_a)
                    trial_sim_b = np.array(trial_sim_b)
                    
                    all_sim_a.extend(trial_sim_a)
                    all_sim_b.extend(trial_sim_b)
                    
                    trial_margin = trial_sim_a.mean() - trial_sim_b.mean()
                    if trial_margin > 0:
                        correct_trials += 1
                    total_trials += 1
            
            if len(all_sim_a) == 0:
                print(f"Lag {lag_ms:5.1f}ms: No valid windows found.")
                continue
                
            all_sim_a = np.array(all_sim_a)
            all_sim_b = np.array(all_sim_b)
            margins = all_sim_a - all_sim_b
            
            mean_sim_a = all_sim_a.mean()
            mean_sim_b = all_sim_b.mean()
            mean_delta = mean_sim_a - mean_sim_b
            mean_margin = margins.mean()
            window_acc = (margins > 0).mean()
            trial_acc = correct_trials / total_trials if total_trials > 0 else 0
            
            print(f"Lag {lag:3d} smp | {lag_ms:6.1f} ms | WinAcc: {window_acc*100:5.1f}% | TrAcc: {trial_acc*100:5.1f}% | Marg: {mean_margin:7.4f} | Delta: {mean_delta:7.4f} | SimA: {mean_sim_a:6.4f} | SimB: {mean_sim_b:6.4f}")
            
            results.append({
                'Lag_Samples': lag,
                'Lag_ms': lag_ms,
                'Window_Accuracy': window_acc,
                'Trial_Accuracy': trial_acc,
                'Mean_Margin': mean_margin,
                'Mean_Delta': mean_delta,
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
    
    # 2. Mean Margin & Delta Plot
    plt.figure(figsize=(10, 6))
    plt.plot(lags, df['Mean_Margin'], marker='o', linewidth=2, color='green', label='Mean Margin')
    plt.plot(lags, df['Mean_Delta'], marker='x', linewidth=2, color='purple', linestyle='dashed', label='Mean Delta')
    plt.axhline(0, color='gray', linestyle='dashed', alpha=0.7)
    plt.axvline(0, color='gray', linestyle='dashed', alpha=0.7)
    plt.title("Temporal Lag vs Similarity Margin")
    plt.xlabel("Lag (ms) [Positive = EEG lags Audio]")
    plt.ylabel("Margin / Delta (Cosine Similarity)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("analysis/figures/lag_sweep/lag_vs_margin.png")
    
    # 3. SimA and SimB Plot
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
