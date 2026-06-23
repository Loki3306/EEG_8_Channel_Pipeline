import scipy.io as sio
import numpy as np

def main():
    mat_path = "/kaggle/input/datasets/lowk1ee/s1-klu/S1_KLU.mat"
    print(f"Loading MAT file: {mat_path}")
    try:
        mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    except Exception as e:
        print(f"Error loading {mat_path}: {e}")
        return
        
    if 'trials' in mat:
        trials = mat['trials']
    elif 'trial' in mat:
        trials = mat['trial']
    else:
        print("Could not find 'trials' variable.")
        return
        
    print("\n" + "="*50)
    print("KUL STIMULI DECODING AUDIT")
    print("="*50)
    
    for i in range(min(10, len(trials))):
        trial = trials[i]
        
        att_ear = getattr(trial, 'attended_ear', None)
        att_track = getattr(trial, 'attended_track', None)
        stimuli = getattr(trial, 'stimuli', None)
        
        print(f"\n--- Trial {i} ---")
        print(f"attended_ear   : {att_ear}")
        print(f"attended_track : {att_track}")
        
        if stimuli is None:
            print("stimuli field not found in trial.")
            continue
            
        print(f"stimuli type   : {type(stimuli)}")
        
        if isinstance(stimuli, np.ndarray):
            print(f"stimuli shape  : {stimuli.shape}")
            if stimuli.size >= 2:
                # Need to handle strings wrapped in mat_struct or just plain strings
                left = stimuli[0]
                right = stimuli[1]
                
                print(f"LEFT (stim[0]) : {left}")
                print(f"RIGHT(stim[1]) : {right}")
                
                if str(att_track) == '1':
                    print(f"-> Attended Track = 1. Attended Audio = {left}")
                elif str(att_track) == '2':
                    print(f"-> Attended Track = 2. Attended Audio = {right}")
                else:
                    print(f"-> Attended Track = {att_track}. Mapping unknown.")
            else:
                print(f"stimuli content: {stimuli}")
        else:
            print(f"stimuli value  : {stimuli}")

if __name__ == "__main__":
    main()
