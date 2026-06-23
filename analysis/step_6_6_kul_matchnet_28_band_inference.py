import os
import sys
import argparse
import scipy.io as sio
import scipy.io.wavfile as wavfile
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import resample, butter, filtfilt

# Add repository root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.matchnet import ContrastiveMatchNet
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

def extract_28_band_envelope(audio_data, fs_in, fs_out=64, num_bands=28):
    cfs = erb_space(50, 8000, num_bands)
    lows, highs = get_erb_bands(cfs)
    
    envelopes = []
    for i in range(num_bands):
        band_audio = apply_bandpass(audio_data, lows[i], highs[i], fs_in)
        env = np.abs(band_audio)
        env = env ** 0.3
        envelopes.append(env)
        
    envelopes = np.array(envelopes)
    num_samples_out = int(envelopes.shape[1] * fs_out / fs_in)
    return resample(envelopes, num_samples_out, axis=1)

# --- Normalization ---
def normalize_array(arr):
    arr = arr - arr.mean(axis=0, keepdims=True)
    scale = arr.std(axis=0, keepdims=True) + 1e-12
    return arr / scale

def pearson_corr(x, y, dim=1):
    x_centered = x - x.mean(dim=dim, keepdim=True)
    y_centered = y - y.mean(dim=dim, keepdim=True)
    cov = (x_centered * y_centered).sum(dim=dim)
    var_x = (x_centered ** 2).sum(dim=dim)
    var_y = (y_centered ** 2).sum(dim=dim)
    return cov / torch.sqrt(var_x * var_y + 1e-8)

def find_checkpoint(root_dir):
    candidates = []
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.endswith('.pth') or f.endswith('.pt'):
                candidates.append(os.path.join(root, f))
    for c in candidates:
        if 'best' in c.lower() or 'matchnet' in c.lower(): return c
    return candidates[0] if candidates else None

def main():
    print("\n" + "="*50)
    print("PHASE KUL-3: CONTRASTIVEMATCHNET 28-BAND INFERENCE")
    print("="*50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Checkpoint
    checkpoint_dir = "/kaggle/working/EEG_8_Channel_Pipeline/checkpoints"
    if not os.path.exists(checkpoint_dir):
        checkpoint_dir = "checkpoints" # fallback
        
    chk_path = find_checkpoint(checkpoint_dir)
    if not chk_path:
        # Search parent or root
        chk_path = find_checkpoint(".")
        
    if not chk_path:
        print("ERROR: No DTU MatchNet checkpoint found!")
        return
        
    print(f"Loading DTU Checkpoint: {chk_path}")
    model = ContrastiveMatchNet(eeg_model_type="eegnet", eeg_channels=8, audio_channels=28, latent_dim=64).to(device)
    model.load_state_dict(torch.load(chk_path, map_location=device))
    model.eval()
    
    # 2. Generate Tensors
    mat_path = "/kaggle/input/datasets/lowk1ee/s1-klu/S1_KLU.mat"
    wav_dir = "/kaggle/input/datasets/lowk1ee/audio-klu"
    
    print("\nExtracting KUL Trial 0 and generating 28-band envelopes...")
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    trials = mat['trials'] if 'trials' in mat else mat['trial']
    trial = trials[0]
    
    eeg_data = trial.RawData.EegData
    fs_eeg = trial.FileHeader.SampleRate
    channel_names = [ch.Label for ch in trial.FileHeader.Channels]
    target_channels = ['Fp1', 'Fp2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8']
    selected_indices = [channel_names.index(tc) if tc in channel_names else [c.upper() for c in channel_names].index(tc.upper()) for tc in target_channels]
    
    eeg_8 = eeg_data[:, selected_indices]
    fs_dtu = 64
    # Bandpass EEG
    nyq = 0.5 * fs_eeg
    b, a = butter(2, [1.0/nyq, 6.0/nyq], btype='band')
    eeg_8 = filtfilt(b, a, eeg_8, axis=0)
    
    eeg_64 = resample(eeg_8, int(len(eeg_8) * fs_dtu / fs_eeg), axis=0)
    
    att_ear = trial.attended_ear
    stimuli = trial.stimuli
    att_wav_name = stimuli[0] if att_ear == 'L' else stimuli[1]
    unatt_wav_name = stimuli[1] if att_ear == 'L' else stimuli[0]
    
    def find_wav(name):
        for r, d, f in os.walk(wav_dir):
            if name in f: return os.path.join(r, name)
            if name+".wav" in f: return os.path.join(r, name+".wav")
        return None
        
    fs_att, audio_att = wavfile.read(find_wav(str(att_wav_name)))
    fs_unatt, audio_unatt = wavfile.read(find_wav(str(unatt_wav_name)))
    
    if len(audio_att.shape) > 1: audio_att = audio_att.mean(axis=1)
    if len(audio_unatt.shape) > 1: audio_unatt = audio_unatt.mean(axis=1)
    
    env_att = extract_28_band_envelope(audio_att, fs_att, fs_out=64, num_bands=28)
    env_unatt = extract_28_band_envelope(audio_unatt, fs_unatt, fs_out=64, num_bands=28)
    
    # 3. Align and Normalize
    min_len = min(len(eeg_64), env_att.shape[1], env_unatt.shape[1])
    eeg_64 = eeg_64[:min_len]
    env_att = env_att[:, :min_len]
    env_unatt = env_unatt[:, :min_len]
    
    # Normalize globally like DTU
    eeg_norm = normalize_array(eeg_64) # (T, 8)
    env_att = normalize_array(env_att.T).T # (28, T)
    env_unatt = normalize_array(env_unatt.T).T
    
    # 4. Inference loop
    window_sec, stride_sec = 3.0, 1.5
    win_samples = int(window_sec * fs_dtu)
    stride_samples = int(stride_sec * fs_dtu)
    
    print("\nRunning Inference on first 20 Windows...")
    margins = []
    
    with torch.no_grad():
        window_idx = 0
        for start in range(0, min_len - win_samples + 1, stride_samples):
            if window_idx >= 20: break
            
            x = eeg_norm[start:start+win_samples].T # (8, 192)
            ya = env_att[:, start:start+win_samples] # (28, 192)
            yb = env_unatt[:, start:start+win_samples]
            
            x_t = torch.FloatTensor(x).unsqueeze(0).to(device)
            ya_t = torch.FloatTensor(ya).unsqueeze(0).to(device)
            yb_t = torch.FloatTensor(yb).unsqueeze(0).to(device)
            
            z_eeg, z_a, z_b = model(x_t, ya_t, yb_t)
            
            # Use Pearson (this is what export_subject_distance.py uses)
            sim_a = pearson_corr(z_eeg, z_a, dim=1).mean().item()
            sim_b = pearson_corr(z_eeg, z_b, dim=1).mean().item()
            
            margin = abs(sim_a - sim_b)
            margins.append(margin)
            pred = "Attended" if sim_a > sim_b else "Unattended"
            
            norm_eeg = torch.norm(z_eeg, p=2, dim=1).mean().item()
            norm_a = torch.norm(z_a, p=2, dim=1).mean().item()
            norm_b = torch.norm(z_b, p=2, dim=1).mean().item()
            
            print(f"Win {window_idx:02d} | Sim A: {sim_a: 6.3f} | Sim B: {sim_b: 6.3f} | Margin: {margin:.4f} | Pred: {pred:<10} | ||Ze||: {norm_eeg:.2f} ||Za||: {norm_a:.2f} ||Zb||: {norm_b:.2f}")
            
            window_idx += 1
            
    margins = np.array(margins)
    print("\n5. Distribution Audit (First 20 windows KUL)")
    print(f"Mean Margin: {margins.mean():.4f}")
    print(f"Std Margin : {margins.std():.4f}")
    print(f"Min Margin : {margins.min():.4f}")
    print(f"Max Margin : {margins.max():.4f}")
    
    print("\nFINAL VERDICT: A. Forward pass successful")

if __name__ == "__main__":
    main()
