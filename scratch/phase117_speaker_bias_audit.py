import os
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import welch
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score

# First, find which channel is Male for each of the 60 trials
wav_dir = '/kaggle/input/datasets/lokeshgile/aasd-audio/Stimuli Audio'
if not os.path.exists(wav_dir):
    wav_dir = '/kaggle/input/aasd-audio/Stimuli Audio'

def detect_male_channel(wav_path):
    sr, data = wav.read(wav_path)
    f_L, Pxx_L = welch(data[:, 0], sr, nperseg=sr)
    f_R, Pxx_R = welch(data[:, 1], sr, nperseg=sr)
    valid_idx = (f_L >= 50) & (f_L <= 300)
    peak_L = f_L[valid_idx][np.argmax(Pxx_L[valid_idx])]
    peak_R = f_R[valid_idx][np.argmax(Pxx_R[valid_idx])]
    return 'L' if peak_L < peak_R else 'R'

print("Detecting Male/Female spatial assignment for 60 trials...")
male_assignment = {}
for i in range(1, 61):
    wav_path = os.path.join(wav_dir, f"mixed_{i:03d}.wav")
    if os.path.exists(wav_path):
        male_assignment[i] = detect_male_channel(wav_path)
        
print(f"Successfully detected gender assignment for {len(male_assignment)} trials.")

# Now let's do a mock pass through the cache to see if subjects are selectively attending Male or Female!
cache_dir = Path('/kaggle/working/multiband_cache')
possible_paths = [
    Path('/kaggle/input/datasets/lowkieee/multiband-cache/kaggle/working/multiband_cache'),
    Path('/kaggle/input/multiband-cache/kaggle/working/multiband_cache'),
    Path('/kaggle/input/aasd-universal-cache-v1/kaggle/working/multiband_cache'),
    cache_dir
]
for p in possible_paths:
    if p.exists() and len(list(p.glob('*_multiband.pt'))) > 0:
        cache_dir = p
        break

cache_files = sorted(list(cache_dir.glob('*_multiband.pt')))

print("\n=======================================================")
print(" SPEAKER BIAS AUDIT (MALE VS FEMALE ATTENTION)")
print("=======================================================")
for cache_file in cache_files:
    subj_name = cache_file.stem.split('_')[0]
    cached = torch.load(cache_file, map_location='cpu', weights_only=False)['raw']
    
    attend_male_count = 0
    attend_female_count = 0
    
    for trial_idx, tr in enumerate(cached):
        audio_id = trial_idx + 1 # Assuming strictly ordered 1-60
        if audio_id not in male_assignment: continue
            
        male_ch = male_assignment[audio_id]
        
        # Get dominant attended speaker
        sp = tr['meta']['switch_points']
        T = tr['eeg'].shape[1]
        boundaries = [0] + [idx for spk, idx in sp]
        boundaries = sorted(set(boundaries))
        if boundaries[-1] != T: boundaries.append(T)
            
        l_duration = 0
        r_duration = 0
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i+1]
            current_spk = 'L'
            for spk, idx in sp:
                if idx <= start_idx: current_spk = spk
                else: break
            if current_spk == 'L': l_duration += (end_idx - start_idx)
            else: r_duration += (end_idx - start_idx)
            
        dom_spk = 'L' if l_duration >= r_duration else 'R'
        
        if dom_spk == male_ch:
            attend_male_count += 1
        else:
            attend_female_count += 1
            
    total = attend_male_count + attend_female_count
    if total > 0:
        pct_male = 100.0 * attend_male_count / total
        print(f"{subj_name:<10} Attended Male: {attend_male_count:<2} ({pct_male:.1f}%) | Attended Female: {attend_female_count:<2}")
