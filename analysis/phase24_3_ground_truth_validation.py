import scipy.io
import argparse
import numpy as np
import os
import glob
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

def validate_and_generate(data_dir, output_file):
    mat_files = glob.glob(os.path.join(data_dir, '*', '*.mat'))
    if not mat_files:
        print(f"No .mat files found in {data_dir}")
        return
        
    fs = 128.0
    trial_length = 7680
    
    # Validation stats
    stats = {
        'total_trials': 0,
        'missing_initial_lock': 0,
        'invalid_event_codes': 0,
        'redundant_presses': 0,
        'simultaneous_events': 0,
        'missing_audio_marker': 0
    }
    
    # Behavioral stats
    behavior = {
        'switches_per_trial': [],
        'switch_intervals': [],
        'time_left_s': 0.0,
        'time_right_s': 0.0
    }
    
    ground_truth_dict = {}
    
    print(f"Starting audit across {len(mat_files)} subjects...")
    
    for mat_path in mat_files:
        subject_id = os.path.basename(os.path.dirname(mat_path))
        try:
            mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
            eeg_var = [k for k in mat.keys() if not k.startswith('__')][0]
            events = mat[eeg_var].event
        except Exception as e:
            print(f"Error loading {mat_path}: {e}")
            continue
            
        trials = defaultdict(list)
        
        # Parse events
        for i in range(events.shape[0]):
            ev = events[i] if events.ndim > 1 else events
            if events.ndim > 1 and len(ev) >= 5:
                ev_type = str(ev[0]).strip()
                latency = float(ev[1])
                epoch = int(ev[4])
            else:
                ev_type = str(getattr(ev, 'type', '')).strip()
                latency = float(getattr(ev, 'latency', 0))
                epoch = int(getattr(ev, 'epoch', 0))
                
            if ev_type:
                trials[epoch].append((ev_type, latency))
                
        for epoch, epoch_events in trials.items():
            stats['total_trials'] += 1
            
            # Sort chronologically
            epoch_events.sort(key=lambda x: x[1])
            
            # Check F: Audio marker
            first_event = epoch_events[0]
            if first_event[0] in ['179', '184']:
                stats['missing_audio_marker'] += 1
                audio_marker = "Unknown"
                trial_start = first_event[1]
            else:
                audio_marker = first_event[0]
                trial_start = first_event[1]
                
            attn_events = [(t, l) for t, l in epoch_events if t in ['179', '184']]
            invalid_events = [t for t, l in epoch_events if t not in ['179', '184', audio_marker]]
            stats['invalid_event_codes'] += len(invalid_events)
            
            # Check A & D: Initial lock
            if len(attn_events) == 0:
                stats['missing_initial_lock'] += 1
                continue
                
            first_lock_time = (attn_events[0][1] - trial_start) / fs
            if first_lock_time > 5.0:
                stats['missing_initial_lock'] += 1
                
            # Check E: Simultaneous
            for i in range(1, len(attn_events)):
                if attn_events[i][1] - attn_events[i-1][1] < 10: # < 80ms
                    stats['simultaneous_events'] += 1
                    
            # Build dense vector (NaN until first lock)
            dense_gt = np.full(trial_length, np.nan, dtype=np.float32)
            
            current_state = None
            last_switch_latency = trial_start
            valid_switches = 0
            
            for ev_type, latency in attn_events:
                state_val = 0.0 if ev_type == '179' else 1.0
                
                # Check C: Redundant presses
                if current_state == state_val:
                    stats['redundant_presses'] += 1
                else:
                    if current_state is not None:
                        # True cognitive shift
                        interval = (latency - last_switch_latency) / fs
                        behavior['switch_intervals'].append(interval)
                        
                        if current_state == 0.0:
                            behavior['time_left_s'] += interval
                        else:
                            behavior['time_right_s'] += interval
                            
                        valid_switches += 1
                        
                    last_switch_latency = latency
                    current_state = state_val
                    
                # Fill array from this latency
                idx = int(latency - trial_start)
                idx = max(0, min(idx, trial_length-1))
                dense_gt[idx:] = current_state
                
            # Add final segment
            final_interval = (trial_start + trial_length - last_switch_latency) / fs
            if current_state == 0.0:
                behavior['time_left_s'] += final_interval
            elif current_state == 1.0:
                behavior['time_right_s'] += final_interval
                
            behavior['switches_per_trial'].append(valid_switches)
            
            key = f"{subject_id}_trial{epoch}_audio{audio_marker}"
            ground_truth_dict[key] = dense_gt

    print("\n====================================================")
    print("PHASE 24.3: EVENT SEMANTICS & GROUND TRUTH VALIDATION")
    print("====================================================")
    
    print("\n--- 1. ASSUMPTION VALIDATION ---")
    print(f"Total Trials Audited       : {stats['total_trials']}")
    print(f"[A/D] Missing Initial Lock : {stats['missing_initial_lock']}  (>5s or absent)")
    print(f"[B] Invalid Event Codes    : {stats['invalid_event_codes']}")
    print(f"[C] Redundant Presses      : {stats['redundant_presses']}  (e.g., L->L)")
    print(f"[E] Simultaneous Events    : {stats['simultaneous_events']}  (<80ms gap)")
    print(f"[F] Missing Audio Marker   : {stats['missing_audio_marker']}")
    
    print("\n--- 2. BEHAVIORAL STATISTICS ---")
    switches = behavior['switches_per_trial']
    intervals = behavior['switch_intervals']
    if switches:
        print(f"Switches/Trial - Mean: {np.mean(switches):.2f}, Median: {np.median(switches):.2f}, Std: {np.std(switches):.2f}")
        print(f"Switches/Trial - Min: {np.min(switches)}, Max: {np.max(switches)}")
    if intervals:
        print(f"Switch Intervals - Mean: {np.mean(intervals):.2f}s, Median: {np.median(intervals):.2f}s")
        
    tot_time = behavior['time_left_s'] + behavior['time_right_s']
    if tot_time > 0:
        left_pct = (behavior['time_left_s'] / tot_time) * 100
        right_pct = (behavior['time_right_s'] / tot_time) * 100
        print(f"Left/Right Bias  - Left: {left_pct:.1f}%, Right: {right_pct:.1f}%")
        
    print("\n--- 3. GROUND TRUTH GENERATION ---")
    print(f"Successfully reconstructed {len(ground_truth_dict)} dense ground-truth vectors.")
    
    if output_file:
        np.savez_compressed(output_file, **ground_truth_dict)
        print(f"Saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to 'Processed EEG' directory")
    parser.add_argument("--output_file", type=str, default="aasd_dense_ground_truth.npz")
    args = parser.parse_args()
    validate_and_generate(args.data_dir, args.output_file)
