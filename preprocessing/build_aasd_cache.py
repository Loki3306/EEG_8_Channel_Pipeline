import os
import sys
import numpy as np
import scipy.io
import scipy.signal
import torch
import glob
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def get_ev_attr(e, attr_name, array_idx=0):
    try:
        if hasattr(e, attr_name):
            return getattr(e, attr_name)
        if isinstance(e, np.ndarray):
            if e.size == 1 and hasattr(e.flat[0], attr_name):
                return getattr(e.flat[0], attr_name)
            return e[array_idx]
    except:
        pass
    return ''

def build_aasd_cache():
    eeg_dir = '/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG'
    audio_dir = '/kaggle/input/datasets/lokeshgile/aasd-audio-gammatones'
    out_dir = REPO_ROOT / 'data' / 'processed_aasd'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    target_channels = ['T7', 'C2', 'FT8', 'P7', 'CPz', 'Fp1', 'TP8', 'C3']
    fallback_map = {'T7': 23, 'C2': 28, 'FT8': 22, 'P7': 41, 'CPz': 36, 'Fp1': 0, 'TP8': 40, 'C3': 25}
    sel_idx = [fallback_map[tc] for tc in target_channels]
    
    b, a = scipy.signal.butter(4, [1.0/32.0, 8.0/32.0], btype='band')
    
    # We will process all available AASD subjects in the directory
    eeg_files = glob.glob(os.path.join(eeg_dir, '*', '*.mat'))
    if not eeg_files:
        print(f"[ERROR] No AASD EEG files found in {eeg_dir}")
        return
        
    print(f"Found {len(eeg_files)} AASD subject files. Building offline cache in {out_dir} ...")

    for filepath in sorted(eeg_files):
        sub_id = os.path.splitext(os.path.basename(filepath))[0].upper()
        save_file = out_dir / f"{sub_id}.pt"
        
        if save_file.exists():
            print(f"[{sub_id}] Cache already exists at {save_file}, skipping.")
            continue
            
        print(f"\nProcessing Subject {sub_id} ...")
        
        mat = scipy.io.loadmat(filepath, squeeze_me=True, struct_as_record=False)
        eeg_var = [k for k in mat.keys() if not k.startswith('__')][0]
        eeg_obj = mat[eeg_var]
        data_all = eeg_obj.data
        events = eeg_obj.event

        if len(data_all.shape) == 3:
            data_all = np.concatenate([data_all[:, :, i] for i in range(data_all.shape[2])], axis=1)
            
        # 1. 64-Channel CAR (mean over channels)
        data_all = data_all - data_all.mean(axis=0, keepdims=True)
            
        eeg_filt = scipy.signal.filtfilt(b, a, data_all, axis=1)
        eeg_64 = scipy.signal.resample_poly(eeg_filt, 1, 2, axis=1)
        eeg_8 = eeg_64[sel_idx, :]

        trial_starts = []
        for i, ev in enumerate(events):
            t_str = str(get_ev_attr(ev, 'type', 0)).strip()
            if t_str and t_str not in ['179', '184', '254', '255']:
                try:
                    lat = float(get_ev_attr(ev, 'latency', 1))
                    trial_starts.append((i, t_str, lat))
                except:
                    pass

        valid_trials = []
        
        for idx_ev, (ev_idx, audio_marker, trial_start_lat) in enumerate(trial_starts):
            npz_path = os.path.join(audio_dir, f"{int(audio_marker)}.npz")
            if not os.path.exists(npz_path):
                print(f"  Missing audio file {npz_path} for trial {idx_ev}")
                continue
                
            audio_data = np.load(npz_path)
            env_l_1d = audio_data['env_l']
            env_r_1d = audio_data['env_r']

            next_start_lat = trial_starts[idx_ev+1][2] if idx_ev+1 < len(trial_starts) else data_all.shape[1]
            if next_start_lat - trial_start_lat < 128 * 10: 
                print(f"  Trial {idx_ev} too short, skipping.")
                continue
                
            start_64 = int(trial_start_lat // 2)
            end_64 = int(next_start_lat // 2)
            trial_eeg_8 = eeg_8[:, start_64:end_64]
            
            # Truncate to match audio
            min_len = min(trial_eeg_8.shape[1], len(env_l_1d))
            if min_len < 64 * 5: # Need at least 5 seconds
                print(f"  Trial {idx_ev} too short after audio matching, skipping.")
                continue
                
            trial_eeg_8 = trial_eeg_8[:, :min_len]
            env_l_1d = env_l_1d[:min_len]
            env_r_1d = env_r_1d[:min_len]
            
            # 2. Z-score EEG over the entire trial
            trial_eeg_8 = trial_eeg_8 - trial_eeg_8.mean(axis=1, keepdims=True)
            trial_eeg_8 = trial_eeg_8 / (trial_eeg_8.std(axis=1, keepdims=True) + 1e-12)
            
            # 3. Z-score Audio over the entire trial
            env_l_1d = (env_l_1d - env_l_1d.mean()) / (env_l_1d.std() + 1e-12)
            env_r_1d = (env_r_1d - env_r_1d.mean()) / (env_r_1d.std() + 1e-12)

            # We need to extract the ground truth attention so we can map it to 'audio_a' (attended) and 'audio_b' (unattended).
            # AASD dataset uses dynamic attention switching, but PyTorch training for MatchNet usually expects 
            # static attention per window.
            # We will save the raw Left/Right audio, and the GT labels, so the training script can chunk them.
            
            raw_evs = []
            for ev in events[ev_idx:]:
                try:
                    lat = float(get_ev_attr(ev, 'latency', 1))
                    if lat >= next_start_lat:
                        break
                    t_str = str(get_ev_attr(ev, 'type', 0)).strip()
                    raw_evs.append((t_str, lat - trial_start_lat))
                except:
                    pass
                    
            # For AASD, we don't have a single "attended_ear" for the whole trial. 
            # We'll save env_l and env_r as audio_l and audio_r, and save raw_evs in meta.
            meta = {
                "TrialID": idx_ev + 1,
                "audio_marker": int(audio_marker),
                "raw_evs": raw_evs
            }
            
            valid_trials.append({
                "meta": meta,
                "eeg": torch.FloatTensor(trial_eeg_8),           # (8, time)
                "audio_l": torch.FloatTensor(env_l_1d[np.newaxis, :]), # (1, time)
                "audio_r": torch.FloatTensor(env_r_1d[np.newaxis, :])  # (1, time)
            })
            
        print(f"[{sub_id}] Processed {len(valid_trials)} valid trials.")
        
        # Save to disk
        data_dict = {
            "subject_id": sub_id,
            "trials": valid_trials
        }
        torch.save(data_dict, save_file)
        print(f"[{sub_id}] Saved cached data to {save_file}")

if __name__ == "__main__":
    build_aasd_cache()
