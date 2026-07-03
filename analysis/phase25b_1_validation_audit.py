import numpy as np
import os
import glob
import scipy.io

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

def generate_gt_state(t_array, raw_evs, target_speaker):
    gt = np.zeros(len(t_array))
    if len(raw_evs) == 0:
        return gt
        
    st_times = []
    types = []
    for ev_t, ev_lat in raw_evs:
        if ev_t in ['179', '184', '254', '255']:
            st_times.append(ev_lat / 128.0)
            if target_speaker == 'A':
                types.append('L' if ev_t in ['179', '254'] else 'R')
            else:
                types.append('R' if ev_t in ['179', '254'] else 'L')
                
    if len(types) == 0:
        return gt
        
    # If the first switch is to 'R' (meaning opposite of target), they must have been listening to target ('L') initially.
    # In test_stable_auroc, 'L' -> 1, 'R' -> -1. Here we use 1 for True, 0 for False.
    current_state = 1 if types[0] == 'R' else 0
    for i, t in enumerate(t_array):
        state = current_state
        for st, s_type in zip(st_times, types):
            if t >= st:
                state = 1 if s_type == 'L' else 0
        gt[i] = state
    return gt

def main():
    print("[INFO] Starting Phase 25B.1 Validation Audit (AASD)")
    
    eeg_dir = '/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG'
    audio_dir = '/kaggle/input/datasets/lokeshgile/aasd-audio-gammatones'

    mat_files = glob.glob(os.path.join(eeg_dir, '*', '*.mat'))[:3]
    
    total_expected_raw = 0
    total_trials = 0
    total_valid_trials = 0
    total_gt_array_switches = 0
    
    print("\n--- Subject Breakdown ---")
    
    for mf in mat_files:
        subj = os.path.basename(mf)
        mat = scipy.io.loadmat(mf, squeeze_me=True, struct_as_record=False)
        eeg_var = [k for k in mat.keys() if not k.startswith('__')][0]
        eeg_obj = mat[eeg_var]
        events = eeg_obj.event
        data_all = eeg_obj.data

        # 1. Count switches directly from AASD event files
        raw_switches = 0
        for ev in events:
            t_str = str(get_ev_attr(ev, 'type', 0)).strip()
            if t_str in ['179', '184', '254', '255']:
                raw_switches += 1
                
        total_expected_raw += raw_switches

        # Find boundaries
        trial_starts = []
        for i, ev in enumerate(events):
            t_str = str(get_ev_attr(ev, 'type', 0)).strip()
            if t_str and t_str not in ['179', '184', '254', '255']:
                try:
                    lat = float(get_ev_attr(ev, 'latency', 1))
                    trial_starts.append((i, t_str, lat))
                except:
                    pass
                    
        total_trials += len(trial_starts)

        subj_valid_trials = 0
        subj_gt_array_switches = 0

        for idx_ev, (ev_idx, audio_marker, trial_start_lat) in enumerate(trial_starts):
            npz_path = os.path.join(audio_dir, f"{int(audio_marker)}.npz")
            if not os.path.exists(npz_path):
                continue
                
            audio_data = np.load(npz_path)
            env_l_1d = audio_data['env_l']
            
            next_start_lat = trial_starts[idx_ev+1][2] if idx_ev+1 < len(trial_starts) else data_all.shape[1]
            if next_start_lat - trial_start_lat < 128 * 10: 
                continue
                
            subj_valid_trials += 1
                
            raw_evs = []
            for ev in events[ev_idx:]:
                try:
                    lat = float(get_ev_attr(ev, 'latency', 1))
                    if lat >= next_start_lat:
                        break
                    t_str = str(get_ev_attr(ev, 'type', 0)).strip()
                    if t_str in ['179', '184', '254', '255']:
                        raw_evs.append((t_str, lat - trial_start_lat))
                except:
                    pass

            start_64 = int(trial_start_lat // 2)
            end_64 = int(next_start_lat // 2)
            
            win_len = 128
            hop = 64
            t_array = np.arange(0, min((end_64 - start_64), len(env_l_1d)) - win_len, hop) / 64.0 + 1.0
            
            gt_B = generate_gt_state(t_array, raw_evs, 'B')
            switches_in_array = np.sum(np.abs(np.diff(gt_B)) > 0)
            subj_gt_array_switches += switches_in_array
            
        total_valid_trials += subj_valid_trials
        total_gt_array_switches += subj_gt_array_switches
        
        print(f"Subject: {subj} | Raw Expected: {raw_switches} | Trials: {subj_valid_trials} | GT Array Switches: {subj_gt_array_switches}")

    print("\n==================================================")
    print("PHASE 25B.1 VALIDATION")
    print("==================================================")
    print(f"Subjects               : {len(mat_files)}")
    print(f"Total Trials Found     : {total_trials}")
    print(f"Valid Trials Processed : {total_valid_trials}")
    print(f"Raw Events in MAT      : {total_expected_raw}")
    print(f"Ground Truth Switches  : {total_gt_array_switches}")
    
    if total_gt_array_switches < total_expected_raw * 0.8:
        loss_source = "Trace Generation / Event Parsing Drop"
        trusted = "NO"
    else:
        loss_source = "Controller Metric Calculation Bug (concat issue?)"
        trusted = "YES (but metrics function is bugged)"
        
    print(f"Loss Source            : {loss_source}")
    print(f"Can metrics be trusted?: {trusted}")
    print("==================================================")

if __name__ == "__main__":
    main()
