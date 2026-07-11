import torch
from pathlib import Path

def main():
    cache_dir = Path('/kaggle/input/datasets/lowkieee/multiband-cache/kaggle/working/multiband_cache')
    if not cache_dir.exists():
        cache_dir = Path('./multiband_cache')
        
    possible_paths = [
        Path('/kaggle/input/datasets/lowkieee/multiband-cache/kaggle/working/multiband_cache'),
        Path('/kaggle/input/multiband-cache/kaggle/working/multiband_cache'),
        Path('/kaggle/input/aasd-universal-cache-v1/kaggle/working/multiband_cache'),
        Path('./multiband_cache')
    ]
    
    cache_file = None
    for p in possible_paths:
        if p.exists() and len(list(p.glob('*_multiband.pt'))) > 0:
            cache_file = list(p.glob('*_multiband.pt'))[0]
            break
            
    if cache_file is None:
        print("COULD NOT FIND CACHE DIRECTORY.")
        return
        
    d = torch.load(cache_file, map_location='cpu', weights_only=False)
    
    print("\n--- Trial 0 Metadata ---")
    print(d['raw'][0]['meta'])
    
    print("\n--- Trial 1 Metadata ---")
    print(d['raw'][1]['meta'])

if __name__ == '__main__':
    main()
