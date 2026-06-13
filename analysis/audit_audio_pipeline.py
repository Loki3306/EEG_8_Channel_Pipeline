import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt, resample
from scipy.io import wavfile
import random
from pathlib import Path

def normalize(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-12)

def audit_pipeline():
    print("--- FORENSIC AUDIT: AUDIO PIPELINE ---")
    
    # 1. Load mapping
    print("Loading mapping...")
    with open('data/audio_mapping.json', 'r') as f:
        mapping = json.load(f)
        
    # 2. Load original DTU data
    print("Loading S1_data_preproc.pkl...")
    try:
        with open('data/S1_data_preproc.pkl', 'rb') as f:
            subject_data = pickle.load(f)
    except FileNotFoundError:
        print("ERROR: data/S1_data_preproc.pkl not found. Please run this script in the Kaggle working directory.")
        return
        
    print(f"Loaded {len(subject_data)} trials.")
    
    # 3. Select 5 random trials
    random.seed(42)
    sample_trials = random.sample(subject_data, 5)
    
    correlations = []
    first = True
    
    for i, trial in enumerate(sample_trials):
        trial_id = trial.trial
        story_a = trial.story_a
        story_b = trial.story_b
        
        # Find mapping
        found = None
        for k, v in mapping.items():
            if v['trial'] == trial_id and v['story_a'] == story_a and v['story_b'] == story_b:
                found = v
                break
                
        if not found:
            print(f"Trial {trial_id} ({story_a}/{story_b}) not found in mapping!")
            continue
            
        wav_path = found['wavA']
        if not Path(wav_path).exists():
            print(f"WAV file not found at {wav_path}")
            continue
            
        fs, audio = wavfile.read(wav_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1) # mix to mono
            
        # --- RECONSTRUCT BROADBAND ENVELOPE ---
        # 1. Hilbert envelope
        env = np.abs(hilbert(audio))
        
        # 2. Lowpass at 8Hz (as typical in AAD)
        b, a = butter(3, 8 / (fs / 2), btype='low')
        env_lp = filtfilt(b, a, env)
        
        # 3. Resample to 64Hz
        target_fs = 64
        num_samples = int(len(env_lp) * target_fs / fs)
        env_resampled = resample(env_lp, num_samples)
        
        # 4. Normalize
        env_recon = normalize(env_resampled)
        
        # --- ORIGINAL DTU ENVELOPE ---
        env_dtu = normalize(trial.env_a)
        
        # Align lengths
        min_len = min(len(env_recon), len(env_dtu))
        env_recon = env_recon[:min_len]
        env_dtu = env_dtu[:min_len]
        
        # Compute correlation
        corr = np.corrcoef(env_recon, env_dtu)[0, 1]
        correlations.append(corr)
        print(f"Trial {trial_id} | Orig Length: {len(env_dtu)} | Recon Length: {len(env_recon)} | Correlation: {corr:.4f}")
        
        # Plot the first one
        if first:
            plt.figure(figsize=(15, 5))
            # Plot 10 seconds (640 samples)
            plt.plot(env_dtu[:640], label='Original DTU wavA Envelope', linewidth=2, alpha=0.8)
            plt.plot(env_recon[:640], label='Reconstructed Hilbert Envelope', linewidth=2, alpha=0.8, linestyle='--')
            plt.title(f"Audio Envelope Alignment (First 10s) | Pearson r = {corr:.3f}")
            plt.xlabel("Samples (64Hz)")
            plt.ylabel("Normalized Amplitude")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('envelope_audit.png')
            print("-> Saved plot to envelope_audit.png")
            first = False
            
    print("\n--- AUDIT RESULTS ---")
    print(f"Mean Correlation: {np.mean(correlations):.4f}")
    if np.mean(correlations) < 0.95:
        print("\nWARNING: Correlation is below 0.95!")
        print("This typically happens because:")
        print("1. ALIGNMENT/DELAY: The DTU dataset applied a specific neurophysiological delay (e.g. 100ms) or shifted the audio relative to the EEG triggers.")
        print("2. EXTRACTION METHOD: DTU does NOT use a simple Hilbert transform. They use a Gammatone filterbank (typically 28 bands), followed by a power-law compression (exponent 0.6), and then sum across bands before downsampling.")
        print("3. PADDING/RESAMPLING: The exact boundary conditions or zero-padding used in their `resample` function differ slightly.")
        
if __name__ == "__main__":
    audit_pipeline()
