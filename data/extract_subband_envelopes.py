import os
import pickle
import numpy as np
from pathlib import Path
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, hilbert, resample
import warnings
warnings.filterwarnings("ignore")

AUDIO_DIR = Path(r"C:\Users\lokes\Downloads\AUDIO")
OUT_FILE = Path(__file__).resolve().parents[1] / "data" / "subband_envelopes.pkl"

def get_log_spaced_edges(fmin, fmax, num_bands):
    return np.logspace(np.log10(fmin), np.log10(fmax), num=num_bands + 1)

def extract_subband_envelopes(wav_path, num_bands=8, fmin=100, fmax=8000, target_fs=64):
    fs, data = wavfile.read(wav_path)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1) # mix to mono
        
    edges = get_log_spaced_edges(fmin, fmax, num_bands)
    nyq = 0.5 * fs
    
    bands = []
    
    for i in range(num_bands):
        low = edges[i]
        high = edges[i+1]
        
        # Bandpass filter
        b, a = butter(4, [low / nyq, min(high / nyq, 0.99)], btype='band')
        filtered = filtfilt(b, a, data)
        
        # Hilbert envelope
        analytic = hilbert(filtered)
        envelope = np.abs(analytic)
        
        # Low pass filter at 8 Hz to capture envelope modulations
        b_lp, a_lp = butter(4, 8.0 / nyq, btype='low')
        env_lp = filtfilt(b_lp, a_lp, envelope)
        
        # Resample to 64Hz
        num_samples = int(len(env_lp) * target_fs / fs)
        env_resampled = resample(env_lp, num_samples)
        
        bands.append(env_resampled)
        
    return np.vstack(bands) # shape: (8, Time)

def main(audio_dir):
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    wav_files = list(audio_dir.glob("*.wav"))
    print(f"Extracting 8 sub-band envelopes for {len(wav_files)} files...")
    
    results = {}
    for i, w in enumerate(wav_files):
        if (i+1) % 10 == 0:
            print(f"[{i+1}/{len(wav_files)}] Processing...")
        try:
            env = extract_subband_envelopes(w)
            results[w.name] = env
        except Exception as e:
            print(f"Failed {w.name}: {e}")
            
    with open(OUT_FILE, "wb") as f:
        pickle.dump(results, f)
        
    print(f"Done. Saved to {OUT_FILE}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", type=str, default=r"C:\Users\lokes\Downloads\AUDIO", help="Path to raw WAV files")
    args = parser.parse_args()
    main(Path(args.audio_dir))
