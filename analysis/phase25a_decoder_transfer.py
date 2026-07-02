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
    x_mean = x.mean()
    y_mean = y.mean()
    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
    return num / (den + eps)

def extract_true_gammatone_envelopes(wav_path, target_fs=64):
    """Uses real 28-band ERB gammatone filterbank."""
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
    """Trial-wise Channel-wise Normalization matching KUL."""
    # env is (Channels, Time)
    env = env - env.mean(axis=1, keepdims=True)
    env = env / (env.std(axis=1, keepdims=True) + 1e-12)
    return env

def run_phase25a(aasd_eeg_path, aasd_audio_path, checkpoint_path, out_dir):
    print("====================================================")
    print("PHASE 25A: STRICT PIPELINE EQUIVALENCE")
    print("====================================================")
    
    os.makedirs(out_dir, exist_ok=True)
    
    model = AADConformer(in_channels=8)
    if checkpoint_path and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict, strict=False)
        print(f"[INFO] Loaded Conformer checkpoint: {checkpoint_path}")
    else:
        print("[WARN] Using random weights!")
    model.eval()
    
    print("[INFO] Loading AASD Trial 1...")
    mat = scipy.io.loadmat(aasd_eeg_path, squeeze_me=True, struct_as_record=False)
    eeg_var = [k for k in mat.keys() if not k.startswith('__')][0]
    eeg_data = mat[eeg_var].data
    events = mat[eeg_var].event
    
    trial_eeg_128 = eeg_data[:, :, 0] # (62, 7680) at 128Hz
    
    # 1. EEG Filtering (1-8Hz) matching KUL
    nyq = 128 / 2
    b, a = signal.butter(4, [1.0/nyq, 8.0/nyq], btype='band')
    trial_eeg_128_filt = signal.filtfilt(b, a, trial_eeg_128, axis=1)
    
    # 2. Downsample to 64Hz
    import math
    g = math.gcd(64, 128)
    trial_eeg_64 = signal.resample_poly(trial_eeg_128_filt, 64 // g, 128 // g, axis=1)
    trial_eeg_64 = trial_eeg_64[:8, :] # Keep first 8 channels
    
    # 3. Trial-wise Channel-wise Z-Score
    eeg_norm = norm_env(trial_eeg_64)
    
    audio_marker = None
    for i in range(events.shape[0]):
        ev = events[i] if events.ndim > 1 else events
        ep = int(ev[4] if events.ndim > 1 else getattr(ev, 'epoch', 0))
        if ep == 1:
            ev_t = str(ev[0] if events.ndim > 1 else getattr(ev, 'type', '')).strip()
            if ev_t not in ['179', '184']:
                audio_marker = ev_t
                break
                
    if audio_marker is not None:
        audio_file = os.path.join(aasd_audio_path, f"mixed_{int(audio_marker):03d}.wav")
        if os.path.exists(audio_file):
            print(f"[INFO] Extracting TRUE Gammatone envelopes from {audio_file}...")
            env_l_28, env_r_28 = extract_true_gammatone_envelopes(audio_file, target_fs=64)
            
            # 4. Audio Trial-wise Z-Score (per band)
            env_l_28_norm = norm_env(env_l_28)
            env_r_28_norm = norm_env(env_r_28)
            
            # 5. Subband Pooling
            env_l_1d = env_l_28_norm.mean(axis=0)
            env_r_1d = env_r_28_norm.mean(axis=0)
        else:
            print(f"[FAIL] Audio file not found. Stop.")
            sys.exit(1)
    else:
        print(f"[FAIL] Audio marker not found. Stop.")
        sys.exit(1)
        
    fs = 64
    win_len = int(10.0 * fs)
    stride = int(1.0 * fs)
    
    margins = []
    probabilities = []
    
    print("[INFO] Running 10s sliding window inference with double Z-scoring...")
    for start in range(0, min(eeg_norm.shape[1], len(env_l_1d)) - win_len, stride):
        win_eeg = eeg_norm[:, start:start+win_len]
        
        # 6. Window-level Z-Score (EEG)
        w_eeg_mean = win_eeg.mean(axis=1, keepdims=True)
        w_eeg_std = win_eeg.std(axis=1, keepdims=True) + 1e-8
        win_eeg_norm = (win_eeg - w_eeg_mean) / w_eeg_std
        
        win_eeg_t = torch.tensor(win_eeg_norm, dtype=torch.float32).unsqueeze(0)
        
        win_l = env_l_1d[start:start+win_len]
        win_r = env_r_1d[start:start+win_len]
        
        # 7. Window-level Z-Score (Audio)
        win_l_norm = (win_l - win_l.mean()) / (win_l.std() + 1e-8)
        win_r_norm = (win_r - win_r.mean()) / (win_r.std() + 1e-8)
        
        with torch.no_grad():
            out, _ = model(win_eeg_t, return_features=True)
            pred_env = out.squeeze().numpy()
            
        # 8. Pearson Correlation (exactly as in KUL)
        corr_l = safe_corr_np(pred_env, win_l_norm)
        corr_r = safe_corr_np(pred_env, win_r_norm)
        
        margin = corr_l - corr_r
        prob = 1.0 / (1.0 + np.exp(-10 * margin))
        
        margins.append(margin)
        probabilities.append(prob)
        
    print("[PASS] Inference completed.")
    
    plt.figure(figsize=(15, 10))
    plt.subplot(3, 1, 1)
    plt.plot(margins, label='Margin (Corr L - Corr R)')
    plt.axhline(0, color='r', linestyle='--')
    plt.title('Decoding Margin over Time (10s windows)')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.hist(margins, bins=30, alpha=0.7)
    plt.axvline(0, color='r', linestyle='--')
    plt.title('Margin Histogram')
    
    plt.subplot(3, 1, 3)
    plt.hist(probabilities, bins=30, alpha=0.7, color='green')
    plt.axvline(0.5, color='r', linestyle='--')
    plt.title('Probability Histogram')
    
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
