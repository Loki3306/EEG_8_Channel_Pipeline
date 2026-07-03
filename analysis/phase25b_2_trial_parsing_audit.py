import os
import numpy as np
import pandas as pd
import scipy.io
from scipy import signal
import glob

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

def main():
    print("[INFO] Starting Phase 25B.2 Root Cause Audit")
    
    eeg_dir = '/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG'
    audio_dir = '/kaggle/input/datasets/lokeshgile/aasd-audio-gammatones'

    mat_files = glob.glob(os.path.join(eeg_dir, '*', '*.mat'))[:3]
    if not mat_files:
        print("[ERROR] No MAT files found!")
        return

    for mf in mat_files:
        subj = os.path.basename(mf).replace('.mat', '')
        print(f"\n==================================================")
        print(f"Subject: {subj}")
        print(f"==================================================")
        
        mat = scipy.io.loadmat(mf, squeeze_me=True, struct_as_record=False)
        eeg_var = [k for k in mat.keys() if not k.startswith('__')][0]
        eeg_obj = mat[eeg_var]
        data_all = eeg_obj.data
        events = eeg_obj.event

        if len(data_all.shape) == 3:
            data_all = data_all[:, :, 0]
            
        print(f"Data shape (128Hz): {data_all.shape}")
        
        trial_starts = []
        for i, ev in enumerate(events):
            t_str = str(get_ev_attr(ev, 'type', 0)).strip()
            if t_str and t_str not in ['179', '184', '254', '255']:
                try:
                    lat = float(get_ev_attr(ev, 'latency', 1))
                    trial_starts.append((i, t_str, lat))
                except:
                    pass
                    
        print(f"Total non-switch markers found: {len(trial_starts)}")

        for idx_ev, (ev_idx, audio_marker, trial_start_lat) in enumerate(trial_starts):
            # 1. Event Loading
            npz_path = os.path.join(audio_dir, f"{int(audio_marker)}.npz")
            audio_exists = os.path.exists(npz_path)
            
            # 2. Epoch Extraction
            next_start_lat = trial_starts[idx_ev+1][2] if idx_ev+1 < len(trial_starts) else data_all.shape[1]
            duration_samples = next_start_lat - trial_start_lat
            
            if duration_samples < 128 * 10:
                print(f"Trial {idx_ev:02d} [SKIPPED] - Short duration ({duration_samples/128.0:.1f}s) | marker='{audio_marker}'")
                continue
                
            if not audio_exists:
                print(f"Trial {idx_ev:02d} [SKIPPED] - Missing audio '{int(audio_marker)}.npz'")
                continue
                
            # Simulate processing
            audio_data = np.load(npz_path)
            env_l_1d = audio_data['env_l']
            
            start_64 = int(trial_start_lat // 2)
            end_64 = int(next_start_lat // 2)
            
            # Compute exactly what the benchmark script does
            trial_eeg_shape_1 = end_64 - start_64
            
            win_len = 128
            hop = 64
            
            # Here is the formula from benchmark script
            min_val = min(trial_eeg_shape_1, len(env_l_1d))
            t_array_len = len(np.arange(0, min_val - win_len, hop))
            
            print(f"Trial {idx_ev:02d} [VALID]   - marker='{audio_marker}', duration={duration_samples/128.0:04.1f}s, start_64={start_64:06d}, end_64={end_64:06d}, eeg_shape={trial_eeg_shape_1}, audio_len={len(env_l_1d)}, min_val={min_val}, windows={t_array_len}")

if __name__ == "__main__":
    main()
