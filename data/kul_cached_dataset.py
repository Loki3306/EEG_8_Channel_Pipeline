import torch
from pathlib import Path
import numpy as np
import time

class KULCachedLoader:
    def __init__(self, cache_dir="data/processed_kul"):
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.exists():
            raise FileNotFoundError(f"Cache directory {cache_dir} not found. Please run preprocessing/build_kul_cache.py first.")
            
        self.subjects_data = {}
        
    def load_all(self):
        """Loads all S*.pt files into RAM."""
        start_time = time.time()
        pt_files = list(self.cache_dir.glob("S*.pt"))
        
        if not pt_files:
            raise FileNotFoundError(f"No .pt files found in {self.cache_dir}")
            
        print(f"Loading {len(pt_files)} cached subjects from {self.cache_dir}...")
        
        for pt_file in sorted(pt_files):
            data = torch.load(pt_file)
            sub_id = data["subject_id"]
            self.subjects_data[sub_id] = data["trials"]
            
        print(f"Done in {time.time() - start_time:.2f} seconds.")
        return self.subjects_data

def chunk_data(x, ya, yb, window_sec, hop_sec, fs=64):
    """
    Chunks a single trial (x, ya, yb) into overlapping windows.
    Returns lists of numpy arrays (or tensors if input is tensor).
    """
    win_samples = int(window_sec * fs)
    hop_samples = int(hop_sec * fs)
    chunks_x, chunks_ya, chunks_yb = [], [], []
    
    start = 0
    # Assuming x is shape [Channels, Time]
    while start + win_samples <= x.shape[1]:
        end = start + win_samples
        chunks_x.append(x[:, start:end])
        chunks_ya.append(ya[:, start:end])
        chunks_yb.append(yb[:, start:end])
        start += hop_samples
        
    return chunks_x, chunks_ya, chunks_yb
