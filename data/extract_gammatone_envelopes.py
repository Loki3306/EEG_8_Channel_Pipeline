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

OUT_FILE = Path(__file__).resolve().parents[1] / "data" / "gammatone_envelopes.pkl"

def erb_space(low_freq, high_freq, num_bands):
    erb_low = 21.4 * np.log10(4.37 * low_freq / 1000 + 1)
    erb_high = 21.4 * np.log10(4.37 * high_freq / 1000 + 1)
    erb_points = np.linspace(erb_low, erb_high, num_bands)
    cf = (10 ** (erb_points / 21.4) - 1) / 4.37 * 1000
    return cf

def extract_gammatone_envelopes(wav_path, num_bands=28, low_freq=50, high_freq=8000, target_fs=64):
    fs, data = wavfile.read(wav_path)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1) # mix to mono
        
    if high_freq > fs / 2:
        high_freq = fs / 2 - 100
        
    cfs = erb_space(low_freq, high_freq, num_bands)
    
    # Pre-compute low-pass filter for envelope extraction (8 Hz)
    b_lp, a_lp = butter(3, 8 / (fs / 2), btype='low')
    
    audio_float = data.astype(np.float64)
    
    # Define single band processing function for parallelization
    def process_band(cf):
        b_gt, a_gt = gammatone(cf, 'fir', fs=fs)
        filtered = lfilter(b_gt, a_gt, audio_float)
        compressed = np.abs(filtered) ** 0.6
        env_band = filtfilt(b_lp, a_lp, compressed)
        
        # Use resample_poly instead of resample for 10x+ speedup on long audio
        g = math.gcd(target_fs, fs)
        up = target_fs // g
        down = fs // g
        return resample_poly(env_band, up, down)
        
    bands = Parallel(n_jobs=-1, backend="threading")(
        delayed(process_band)(cf) for cf in cfs
    )
        
    return np.vstack(bands) # shape: (28, Time)

def main(audio_dir):
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    wav_files = list(audio_dir.glob("*.wav"))
    print(f"Extracting 28 gammatone sub-band envelopes for {len(wav_files)} files...")
    
    results = {}
    for i, w in enumerate(wav_files):
        if (i+1) % 10 == 0:
            print(f"[{i+1}/{len(wav_files)}] Processing...")
        try:
            env = extract_gammatone_envelopes(w)
            results[w.name] = env
        except Exception as e:
            print(f"Failed {w.name}: {e}")
            
    with open(OUT_FILE, "wb") as f:
        pickle.dump(results, f)
        
    print(f"Done. Saved to {OUT_FILE}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", type=str, default="/kaggle/input/datasets/lokeshgile/eeg-audio", help="Path to raw WAV files")
    args = parser.parse_args()
    main(Path(args.audio_dir))
