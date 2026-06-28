import os
import sys
import torch
import random
from pathlib import Path
import numpy as np
import scipy.io

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.kul_cached_dataset import KULCachedLoader
from models.kuruvila_original import KuruvilaOriginalCNNLSTM

def main():
    print("="*60)
    print("SEMANTIC AUDIT: KUL LABELS VS ORIGINAL KURUVILA")
    print("="*60)
    
    loader = KULCachedLoader(REPO_ROOT / "data" / "processed_kul")
    try:
        all_data = loader.load_all()
    except FileNotFoundError:
        print("Run on Kaggle to access the dataset cache.")
        return
        
    # Flatten all trials
    all_trials = []
    for sub, trials in all_data.items():
        for t in trials:
            t["meta"]["Subject"] = sub
            all_trials.append(t)
            
    print(f"Loaded {len(all_trials)} total trials from cache.")
    
    # Select 5 random trials
    random.seed(42)
    sample_trials = random.sample(all_trials, 5)
    
    model = KuruvilaOriginalCNNLSTM()
    
    for i, t in enumerate(sample_trials):
        meta = t["meta"]
        eeg = t["eeg"]
        ya = t["audio_a"]
        yb = t["audio_b"]
        
        print(f"\n{'='*50}")
        print(f"TRIAL {i+1}")
        print(f"Subject       : {meta.get('Subject')}")
        print(f"Experiment    : {meta.get('experiment')}")
        print(f"Trial ID      : {meta.get('TrialID')}")
        print(f"Attended Ear  : {meta.get('attended_ear')}")
        print(f"attended_track: {meta.get('attended_track')}")
        
        # Determine labels according to train.py
        attended_track = str(meta.get('attended_track'))
        label = 0 if attended_track == '1' else 1
        
        print(f"Final Label   : {label} (0 = Track 1, 1 = Track 2)")
        
        # Track routing
        if attended_track == '1':
            audio_spk1 = ya
            audio_spk2 = yb
        else:
            audio_spk1 = yb
            audio_spk2 = ya
            
        print("\n--- Audio Routing to Model ---")
        if label == 0:
            print("Since Label == 0 (Track 1 Attended):")
            print("  audio_spk1 receives: audio_a (ATTENDED)")
            print("  audio_spk2 receives: audio_b (UNATTENDED)")
        else:
            print("Since Label == 1 (Track 2 Attended):")
            print("  audio_spk1 receives: audio_b (UNATTENDED)")
            print("  audio_spk2 receives: audio_a (ATTENDED)")
            
        print("\n--- Tensor Shapes ---")
        print(f"EEG Shape        : {eeg.shape}")
        print(f"audio_spk1 Shape : {audio_spk1.shape}")
        print(f"audio_spk2 Shape : {audio_spk2.shape}")
        
        # Take just 1 window (192 samples) to pass to model
        if eeg.shape[1] >= 192:
            eeg_win = eeg[:, :192].unsqueeze(0)
            a1_win = audio_spk1[:, :192].unsqueeze(0)
            a2_win = audio_spk2[:, :192].unsqueeze(0)
            
            print(f"\n--- What is fed to the model ---")
            print(f"Batch EEG : {eeg_win.shape}")
            print(f"Batch A1  : {a1_win.shape}")
            print(f"Batch A2  : {a2_win.shape}")
            
            with torch.no_grad():
                out = model(eeg_win, a1_win, a2_win)
                print(f"\nModel Output (Softmax): {out}")
                
                print(f"\nWhat the model is EXPECTED to output:")
                if label == 0:
                    print(f"EXPECTED: tensor([[1.0000, 0.0000]])")
                else:
                    print(f"EXPECTED: tensor([[0.0000, 1.0000]])")
                    
        print(f"{'='*50}\n")
        
if __name__ == "__main__":
    main()
