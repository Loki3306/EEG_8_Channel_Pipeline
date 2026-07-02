import scipy.io
import argparse

def generate_timeline(mat_path):
    try:
        mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    except Exception as e:
        print(f"Failed to load {mat_path}: {e}")
        return
        
    eeg_var = [k for k in mat.keys() if not k.startswith('__')][0]
    events = mat[eeg_var].event
    
    fs = 128.0
    current_trial = 0
    trial_start_sample = 0
    
    print(f"--- EVENT TIMELINE: {mat_path.split('/')[-1]} ---")
    
    for ev in events:
        ev_type = str(getattr(ev, 'type', ''))
        latency = getattr(ev, 'latency', 0)
        
        # In EEGLAB, epoch is sometimes missing if it's continuous data, 
        # but in this dataset, we saw epoch is column 5.
        epoch = getattr(ev, 'epoch', 0)
        
        # New Trial boundary
        if epoch != current_trial:
            current_trial = epoch
            trial_start_sample = latency
            # We assume numeric strings that aren't 179/184 are audio IDs
            audio_id = ev_type if ev_type not in ['179', '184'] else "Unknown"
            
            try:
                audio_num = int(audio_id)
                print(f"\n[Trial {current_trial}] Audio: mixed_{audio_num:03d}.wav")
            except ValueError:
                print(f"\n[Trial {current_trial}] Audio: {audio_id}")
            continue
            
        if ev_type in ['179', '184']:
            time_in_trial = (latency - trial_start_sample) / fs
            state = "LEFT " if ev_type == '179' else "RIGHT"
            
            # If it's within the first 4 seconds, it's an initial lock
            if time_in_trial < 4.0:
                print(f"  {time_in_trial:05.2f}s : INITIAL LOCK -> {state}")
            else:
                print(f"  {time_in_trial:05.2f}s : SWITCH       -> {state}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat_file", type=str, required=True)
    args = parser.parse_args()
    generate_timeline(args.mat_file)
