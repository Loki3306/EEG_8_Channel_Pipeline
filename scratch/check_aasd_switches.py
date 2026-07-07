import torch
from pathlib import Path

def main():
    cache_path = Path('/kaggle/working/eeg_cache/S1_processed.pt')
    if not cache_path.exists():
        print("Error: Cache not found.")
        return
        
    cached = torch.load(cache_path, map_location='cpu', weights_only=False)
    trials = cached['raw']
    
    print("="*50)
    print(" AASD ATTENTION SWITCH POINTS (SUBJECT 1)")
    print("="*50)
    
    SR = 128
    
    for i in range(5):  # Just check the first 5 trials
        switches = trials[i]['meta']['switch_points']
        print(f"\nTrial {i+1} (60 seconds):")
        
        if len(switches) == 0:
            print("  No switches found. Subject attended to the same speaker for all 60 seconds.")
        else:
            for direction, sample_idx in switches:
                time_sec = sample_idx / SR
                print(f"  -> Switch to {direction} Speaker at {time_sec:.3f} seconds")

if __name__ == "__main__":
    main()
