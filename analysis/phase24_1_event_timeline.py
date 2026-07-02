import scipy.io
import argparse
import sys
import os

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
    
    try:
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
                
            if not ev_type:
                continue
                
            if epoch != current_trial:
                current_trial = epoch
                trial_start_sample = latency
                audio_id = ev_type if ev_type not in ['179', '184'] else "Unknown"
                
                try:
                    audio_num = int(float(audio_id))
                    print(f"\n[Trial {current_trial}] Audio: mixed_{audio_num:03d}.wav")
                except ValueError:
                    print(f"\n[Trial {current_trial}] Audio: {audio_id}")
                continue
                
            if ev_type in ['179', '184']:
                time_in_trial = (latency - trial_start_sample) / fs
                state = "LEFT " if ev_type == '179' else "RIGHT"
                
                if time_in_trial < 4.0:
                    print(f"  {time_in_trial:05.2f}s : INITIAL LOCK -> {state}")
                else:
                    print(f"  {time_in_trial:05.2f}s : SWITCH       -> {state}")
                    
    except BrokenPipeError:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat_file", type=str, required=True)
    args = parser.parse_args()
    generate_timeline(args.mat_file)
