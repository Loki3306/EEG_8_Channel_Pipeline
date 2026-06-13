import argparse
import json
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, resample, gammatone, lfilter
from scipy.io import wavfile
import random
from pathlib import Path
import sys

# Ensure baselines can be imported
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from baselines.ridge_aad import load_subject_examples

def normalize(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-12)

def erb_space(low_freq, high_freq, num_bands):
    erb_low = 21.4 * np.log10(4.37 * low_freq / 1000 + 1)
    erb_high = 21.4 * np.log10(4.37 * high_freq / 1000 + 1)
    erb_points = np.linspace(erb_low, erb_high, num_bands)
    cf = (10 ** (erb_points / 21.4) - 1) / 4.37 * 1000
    return cf

def gammatone_envelope(audio, fs, num_bands=28, low_freq=50, high_freq=8000, target_fs=64):
    """
    Biologically-inspired auditory model:
    1. Gammatone filterbank
    2. Power-law compression (0.6)
    3. Envelope extraction (Low-pass)
    4. Summation across bands
    """
    if high_freq > fs / 2:
        high_freq = fs / 2 - 100
        
    cfs = erb_space(low_freq, high_freq, num_bands)
    
    # Pre-compute low-pass filter for envelope extraction
    b_lp, a_lp = butter(3, 8 / (fs / 2), btype='low')
    
    audio_float = audio.astype(np.float64)
    env_sum = np.zeros_like(audio_float)
    
    for cf in cfs:
        # 1. Gammatone filter
        b_gt, a_gt = gammatone(cf, 'iir', fs=fs)
        filtered = lfilter(b_gt, a_gt, audio_float)
        
        # 2. Rectification & Power-law compression
        compressed = np.abs(filtered) ** 0.6
        
        # 3. Envelope Extraction (lowpass)
        env_band = filtfilt(b_lp, a_lp, compressed)
        
        env_sum += env_band
        
    # 4. Downsample
    num_samples = int(len(env_sum) * target_fs / fs)
    env_resampled = resample(env_sum, num_samples)
    
    return env_resampled

def audit_gammatone():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/kaggle/input/datasets/lokeshgile/dataset-eeg')
    parser.add_argument('--audio_dir', type=str, default='/kaggle/input/dtu-audio-files')
    parser.add_argument('--mapping_file', type=str, default='data/audio_mapping.json')
    args = parser.parse_args()

    print("--- FORENSIC AUDIT: GAMMATONE PIPELINE ---")
    
    print(f"Loading mapping from {args.mapping_file}...")
    with open(args.mapping_file, 'r') as f:
        mapping = json.load(f)
        
    mat_path = Path(args.data_dir) / "S1_data_preproc.mat"
    print(f"Loading {mat_path}...")
    try:
        trials = load_subject_examples(mat_path)
    except Exception as e:
        print(f"ERROR: {e}")
        return
        
    print(f"Loaded {len(trials)} trials.")
    
    random.seed(42)
    sample_indices = random.sample(range(len(trials)), min(20, len(trials)))
    
    correlations = []
    
    for idx in sample_indices:
        trial = trials[idx]
        subject = trial.subject
        trial_index = trial.trial_index
        orig_env_a = trial.wav_a
        
        try:
            found = mapping[subject][f"trial_{trial_index}"]
        except KeyError:
            continue
            
        wav_filename = found['wavA']['filename']
        wav_path = Path(args.audio_dir) / wav_filename
        
        if not wav_path.exists():
            print(f"WAV file not found at {wav_path}")
            continue
            
        fs, audio = wavfile.read(wav_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1) # mix to mono
            
        # Reconstruct with Gammatone model
        env_recon = gammatone_envelope(audio, fs)
        
        env_recon = normalize(env_recon)
        env_dtu = normalize(orig_env_a)
        
        min_len = min(len(env_recon), len(env_dtu))
        env_recon = env_recon[:min_len]
        env_dtu = env_dtu[:min_len]
        
        corr = np.corrcoef(env_recon, env_dtu)[0, 1]
        correlations.append(corr)
        print(f"Trial {trial_index:02d} | Corr: {corr:.4f}")
            
    print("\n--- GAMMATONE AUDIT RESULTS ---")
    if not correlations:
        print("No correlations computed.")
        return
    print(f"Mean Correlation:   {np.mean(correlations):.4f}")
    print(f"Median Correlation: {np.median(correlations):.4f}")
    print(f"Min Correlation:    {np.min(correlations):.4f}")
    print(f"Max Correlation:    {np.max(correlations):.4f}")

if __name__ == "__main__":
    audit_gammatone()
