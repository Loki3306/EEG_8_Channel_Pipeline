import torch
from pathlib import Path
import os
import multiprocessing as mp

def main():
    cache_dir = Path('/kaggle/input/datasets/lowkieee/multiband-cache/kaggle/working/multiband_cache')
    if not cache_dir.exists():
        cache_dir = Path('./multiband_cache')
    
    # Try multiple possible locations
    possible_paths = [
        Path('/kaggle/input/datasets/lowkieee/multiband-cache/kaggle/working/multiband_cache'),
        Path('/kaggle/input/multiband-cache/kaggle/working/multiband_cache'),
        Path('/kaggle/input/aasd-universal-cache-v1/kaggle/working/multiband_cache'),
        Path('./multiband_cache'),
        Path('C:/Users/lokes/.gemini/antigravity-ide/brain/0ad5df83-2069-46f1-8ad2-402fd215df23/multiband_cache')
    ]
    
    cache_file = None
    for p in possible_paths:
        if p.exists() and len(list(p.glob('*_multiband.pt'))) > 0:
            cache_file = list(p.glob('*_multiband.pt'))[0]
            break
            
    if cache_file is None:
        print("COULD NOT FIND CACHE DIRECTORY. Run this on Kaggle.")
        return
        
    print(f"Loading {cache_file}...")
    d = torch.load(cache_file, map_location='cpu', weights_only=False)
    
    trials_with_switches = 0
    total_switches = 0
    total_trials = len(d['raw'])
    
    print("\n--- Switch Points per Trial ---")
    for i, tr in enumerate(d['raw']):
        sp = tr['meta'].get('switch_points', [])
        print(f"Trial {i}: {sp}")
        if len(sp) > 1:
            trials_with_switches += 1
        total_switches += len(sp)
            
    print(f"\nTotal Trials: {total_trials}")
    print(f"Trials with >1 switch point: {trials_with_switches}")
    print(f"Total switch events: {total_switches}")

if __name__ == '__main__':
    main()
