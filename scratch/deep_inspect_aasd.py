import scipy.io
import numpy as np
import os

def deep_inspect_aasd():
    eeg_path = "/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG/S1/S1.mat"

    print("="*60)
    print("DEEP INSPECTION OF 'EEG_new'")
    print("="*60)
    
    if not os.path.exists(eeg_path):
        print(f"File not found: {eeg_path}")
        return

    mat = scipy.io.loadmat(eeg_path, simplify_cells=True)
    eeg_new = mat['EEG_new']
    
    print(f"Keys in EEG_new: {list(eeg_new.keys())}")
    
    # Inspect Data
    if 'data' in eeg_new:
        data = eeg_new['data']
        if isinstance(data, np.ndarray):
            print(f"\n[DATA] Shape: {data.shape}, Dtype: {data.dtype}")
            # Usually EEG is (Channels x Time) or (Trials x Channels x Time)
            print(f"       Number of Dimensions: {data.ndim}")
            if data.ndim == 3:
                print(f"       -> Looks like [Trials x Channels x Time]")
            elif data.ndim == 2:
                print(f"       -> Looks like [Channels x Time]")
        else:
            print(f"\n[DATA] Type: {type(data)}")
            
    # Inspect Events
    if 'event' in eeg_new:
        event = eeg_new['event']
        print(f"\n[EVENT] Type: {type(event)}")
        if isinstance(event, np.ndarray):
            print(f"        Length: {len(event)}")
            print("        Preview of first 5 events:")
            for i in range(min(5, len(event))):
                ev = event[i]
                if isinstance(ev, dict):
                    print(f"          Event {i}: {ev}")
                elif isinstance(ev, np.ndarray):
                    print(f"          Event {i}: {ev}")
                else:
                    try:
                        print(f"          Event {i} fields: {ev.dtype.names}")
                        print(f"          Event {i} values: {ev}")
                    except:
                        print(f"          Event {i}: {ev}")
        elif isinstance(event, dict):
            print(f"        Keys: {list(event.keys())}")
            for k, v in event.items():
                if isinstance(v, np.ndarray):
                    print(f"          - {k}: shape {v.shape}")
                    print(f"            First few: {v[:5]}")
                else:
                    print(f"          - {k}: {v}")

deep_inspect_aasd()
