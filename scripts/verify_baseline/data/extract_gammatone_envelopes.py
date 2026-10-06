import os
import pickle
import numpy as np
from pathlib import Path
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, resample, resample_poly, gammatone, lfilter
import math
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]

def resolve_gammatone_output_file(num_bands=8, custom_path=None) -> Path:
    if custom_path:
        p = Path(custom_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    if (os.name != 'nt') and Path("/kaggle/working").exists():
        return Path(f"/kaggle/working/gammatone_{num_bands}band_envelopes.pkl")
    out = REPO_ROOT / "data" / f"gammatone_{num_bands}band_envelopes.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    return out

def erb_space(low_freq, high_freq, num_bands):
    erb_low = 21.4 * np.log10(4.37 * low_freq / 1000 + 1)
    erb_high = 21.4 * np.log10(4.37 * high_freq / 1000 + 1)
    erb_points = np.linspace(erb_low, erb_high, num_bands)
    cf = (10 ** (erb_points / 21.4) - 1) / 4.37 * 1000
    return cf

def extract_gammatone_envelopes(wav_path, num_bands=8, low_freq=100, high_freq=7500, target_fs=64):
    fs, data = wavfile.read(wav_path)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1) # mix to mono
        
    if high_freq >= fs / 2:
        high_freq = fs / 2 - 1.0
        
    cfs = erb_space(low_freq, high_freq, num_bands)
    
    # Pre-compute low-pass filter for envelope extraction (8 Hz Butterworth)
    b_lp, a_lp = butter(3, 8 / (fs / 2), btype='low')
    
    audio_float = data.astype(np.float64)
    
    def process_band(cf):
        b_gt, a_gt = gammatone(cf, 'fir', fs=fs)
        filtered = lfilter(b_gt, a_gt, audio_float)
        # Cortical power-law compression (0.6)
        compressed = np.abs(filtered) ** 0.6
        env_band = filtfilt(b_lp, a_lp, compressed)
        
        # Polyphase resample for high speed
        g = math.gcd(target_fs, fs)
        up = target_fs // g
        down = fs // g
        return resample_poly(env_band, up, down)
        
    bands = Parallel(n_jobs=-1, backend="threading")(
        delayed(process_band)(cf) for cf in cfs
    )
        
    return np.vstack(bands).astype(np.float32) # shape: (num_bands, Time)

import glob

def discover_audio_directory(custom_audio_dir: str = None) -> Path:
    if custom_audio_dir:
        p = Path(custom_audio_dir)
        if p.exists() and len(list(p.glob("*.wav"))) > 0:
            return p
            
    candidates = [
        Path("/kaggle/input/datasets/lokeshgile/eeg-audio"),
        Path("/kaggle/input/eeg-audio"),
        Path("/kaggle/input/EEG_Audio"),
        Path("/kaggle/input/eeg_audio"),
        REPO_ROOT / "data" / "audio",
        REPO_ROOT / "USCAPES" / "data" / "audio",
    ]
    for c in candidates:
        if c.exists() and len(list(c.glob("*.wav"))) > 0:
            return c
            
    if Path("/kaggle/input").exists():
        try:
            for w in glob.glob("/kaggle/input/**/*.wav", recursive=True):
                return Path(w).parent
        except Exception:
            pass
            
    return candidates[0]

def main(audio_dir=None, num_bands=8, output_file=None):
    if audio_dir is None:
        audio_dir = discover_audio_directory()
    else:
        audio_dir = discover_audio_directory(audio_dir)
        
    out_file = resolve_gammatone_output_file(num_bands=num_bands, custom_path=output_file)
    wav_files = sorted(list(audio_dir.glob("*.wav")))
    if not wav_files:
        raise FileNotFoundError(f"No WAV files found in {audio_dir}")
        
    print(f"Extracting {num_bands} Gammatone sub-band envelopes for {len(wav_files)} files from {audio_dir}...")
    
    results = {}
    for i, w in enumerate(wav_files):
        if (i+1) % 10 == 0 or (i+1) == len(wav_files):
            print(f"  [{i+1}/{len(wav_files)}] Extracted: {w.name}")
        try:
            env = extract_gammatone_envelopes(w, num_bands=num_bands)
            results[w.name] = env
        except Exception as e:
            print(f"Failed {w.name}: {e}")
            
    with open(out_file, "wb") as f:
        pickle.dump(results, f)
        
    print(f"Extraction Complete. Saved {len(results)} envelopes to {out_file}")
    return out_file, results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", type=str, default=None, help="Path to raw WAV directory")
    parser.add_argument("--num_bands", type=int, default=8, help="Number of cochlear Gammatone subbands (default: 8)")
    parser.add_argument("--output_file", type=str, default=None, help="Custom output pickle path")
    args = parser.parse_args()
    main(args.audio_dir, args.num_bands, args.output_file)

