import torch
import numpy as np
import scipy.io
import scipy.io.wavfile as wavfile
from scipy.signal import hilbert, resample
import argparse
import sys
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from models.aad_conformer import AADConformer
    from data.extract_gammatone_envelopes import extract_gammatone_envelopes
except ImportError:
    print("Could not import AADConformer or Gammatone Extractor.")
    sys.exit(1)

def extract_true_gammatone_envelopes(wav_path, target_fs=64):
    """Uses the real 28-band ERB gammatone filterbank and averages across bands to create the 1D target."""
    import tempfile
    
    fs, audio = wavfile.read(wav_path)
    # audio is (Samples, 2)
    left = audio[:, 0].astype(np.float32)
    right = audio[:, 1].astype(np.float32)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tl:
        wavfile.write(tl.name, fs, left)
        # Returns (28, Time)
        env_l_28 = extract_gammatone_envelopes(tl.name, target_fs=target_fs)
        os.remove(tl.name)
        
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tr:
        wavfile.write(tr.name, fs, right)
        env_r_28 = extract_gammatone_envelopes(tr.name, target_fs=target_fs)
        os.remove(tr.name)
        
    # KUL Training Target: Mean across the 28 subbands!
    env_l_1d = env_l_28.mean(axis=0)
    env_r_1d = env_r_28.mean(axis=0)
    
    return env_l_1d, env_r_1d

def run_phase25a(aasd_eeg_path, aasd_audio_path, checkpoint_path, out_dir):
    print("====================================================")
    print("PHASE 25A: DECODER TRANSFER EVALUATION")
    print("====================================================")
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Load Model
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
    
    # Load EEG
    print("[INFO] Loading AASD Trial 1...")
    mat = scipy.io.loadmat(aasd_eeg_path, squeeze_me=True, struct_as_record=False)
    eeg_var = [k for k in mat.keys() if not k.startswith('__')][0]
    eeg_data = mat[eeg_var].data
    events = mat[eeg_var].event
    
    # We will process Trial 0
    trial_eeg = eeg_data[:, :, 0] # (62, 7680) at 128Hz
    
    # Extract audio marker for Trial 1 (EEGLAB epochs are 1-indexed)
    audio_marker = None
    for i in range(events.shape[0]):
        ev = events[i] if events.ndim > 1 else events
        ep = int(ev[4] if events.ndim > 1 else getattr(ev, 'epoch', 0))
        if ep == 1:
            ev_t = str(ev[0] if events.ndim > 1 else getattr(ev, 'type', '')).strip()
            if ev_t not in ['179', '184']:
                audio_marker = ev_t
                break
                
    print(f"[INFO] Found Audio Marker: {audio_marker}")
    
    # Load Audio
    if audio_marker is not None:
        audio_file = os.path.join(aasd_audio_path, f"mixed_{int(audio_marker):03d}.wav")
        if os.path.exists(audio_file):
            print(f"[INFO] Extracting TRUE Gammatone envelopes from {audio_file}...")
            env_l, env_r = extract_true_gammatone_envelopes(audio_file, target_fs=64)
        else:
            print(f"[WARN] Audio file not found. Using mock envelopes.")
            env_l = np.random.randn(60*64)
            env_r = np.random.randn(60*64)
    else:
        print(f"[WARN] Could not find Audio marker. Using mock envelopes.")
        env_l = np.random.randn(60*64)
        env_r = np.random.randn(60*64)
        
    # Preprocess EEG (Global Z-Score + Downsample)
    eeg_64 = trial_eeg[:, ::2] # (62, 3840)
    global_mean = np.mean(eeg_data[:, ::2, :])
    global_std = np.std(eeg_data[:, ::2, :])
    
    eeg_norm = (eeg_64 - global_mean) / global_std
    eeg_norm = eeg_norm[:8, :] # Keep 8 channels
    
    # Sliding Window Inference (2s window, 0.5s stride)
    fs = 64
    win_len = int(2.0 * fs)
    stride = int(0.5 * fs)
    
    margins = []
    probabilities = []
    confidences = []
    
    print("[INFO] Running continuous sliding window inference...")
    for start in range(0, eeg_norm.shape[1] - win_len, stride):
        win_eeg = eeg_norm[:, start:start+win_len]
        win_eeg_t = torch.tensor(win_eeg, dtype=torch.float32).unsqueeze(0)
        
        # Audio chunks
        win_a = env_l[start:start+win_len]
        win_b = env_r[start:start+win_len]
        
        with torch.no_grad():
            out, z_pool = model(win_eeg_t, return_features=True)
            pred_env = out.squeeze().numpy()
            
            if hasattr(model, 'confidence_head'):
                conf = model.predict_confidence(z_pool)
                # Simple mock confidence (softmax margin or var)
                confidences.append(float(conf.std()))
                
        # Calculate Pearson Margin
        corr_a = np.corrcoef(pred_env, win_a)[0, 1] if np.std(win_a) > 0 else 0
        corr_b = np.corrcoef(pred_env, win_b)[0, 1] if np.std(win_b) > 0 else 0
        
        margin = corr_a - corr_b
        # Convert margin to probability (mock logistic mapping)
        prob = 1.0 / (1.0 + np.exp(-10 * margin))
        
        margins.append(margin)
        probabilities.append(prob)
        
    print("[PASS] Inference completed.")
    
    # Generate Plots
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(margins, label='Margin (Corr A - Corr B)')
    plt.axhline(0, color='r', linestyle='--')
    plt.title('Decoding Margin over Time')
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
