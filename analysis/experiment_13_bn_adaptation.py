import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.experiment_10_layer_profiler import (
    load_matchnet,
    compute_kul_activations,
    compute_dtu_activations,
    LayerProfiler,
    calculate_frechet_distance
)

def apply_adabn(model, kul_cache_path='data/KUL_Layer_Activations.pkl'):
    """
    Passes KUL data through the model in train() mode to update BatchNorm running stats.
    We only want to update the EEG branch BN layers, as Audio is already perfect.
    """
    print("\n--- Applying AdaBN to EEG Branch ---")
    
    # Set model to eval first
    model.eval()
    
    # We only set EEG BatchNorm layers to train mode
    bn_layers = []
    for m in model.eeg_encoder.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.train()
            # Optional: reset running stats to 0/1 before adaptation
            m.reset_running_stats()
            # Set momentum to None to just use a simple average over the dataset
            m.momentum = None 
            bn_layers.append(m)
            
    print(f"Found {len(bn_layers)} EEG BatchNorm layers to adapt.")
    
    # To pass KUL data through, we need the raw windows. 
    # But wait, compute_kul_activations expects a profiler. 
    # Let's create a dummy profiler that doesn't hook, just passes data forward.
    class DummyProfiler:
        def __init__(self, model, device):
            self.model = model
            self.device = device
            self.activations = {}
        def clear(self): pass
        def get_activations(self): return {}
        
    dummy = DummyProfiler(model, next(model.parameters()).device)
    
    # We call compute_kul_activations just to iterate over KUL and push it through the network
    # The BN layers will update their stats automatically.
    print("Streaming KUL dataset through network to recompute BN statistics...")
    compute_kul_activations(dummy)
    
    # Re-freeze BN
    for m in bn_layers:
        m.eval()
        
    print("AdaBN update complete.")
    return model

def run_adabn_experiment():
    print("="*70)
    print("PHASE D3: ADAPTIVE BATCH NORMALIZATION (AdaBN)")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Original Model & DTU Cache
    model = load_matchnet(device)
    if model is None: return
    
    dtu_cache_path = 'data/DTU_Layer_Activations.pkl'
    if not os.path.exists(dtu_cache_path):
        print("DTU cache missing. Run E10 first.")
        return
        
    with open(dtu_cache_path, 'rb') as f:
        dtu_act = pickle.load(f)
        
    # 2. Apply AdaBN using KUL data
    # (Updates BN running mean/var using KUL distribution)
    adapted_model = apply_adabn(model)
    
    # 3. Profile KUL with Adapted Model
    print("\n--- Profiling KUL with AdaBN Model ---")
    profiler = LayerProfiler(adapted_model, device)
    kul_act_adabn = compute_kul_activations(profiler)
    
    print("\n" + "="*70)
    print("DISTRIBUTION DIVERGENCE (DTU Original vs KUL AdaBN)")
    print("="*70)
    
    layers = [
        "EEG_Block1", "EEG_Block2", "EEG_Embedding",
        "Audio_Conv1", "Audio_Conv2", "Audio_Embedding"
    ]
    
    print(f"{'Layer':<20} | {'Dim':<5} | {'Cosine Dist':<15} | {'Fréchet Distance'}")
    print("-" * 70)
    
    for layer in layers:
        if layer not in dtu_act or layer not in kul_act_adabn: continue
        
        dtu_arr = dtu_act[layer]
        kul_arr = kul_act_adabn[layer]
        
        if len(dtu_arr) == 0 or len(kul_arr) == 0: continue
        
        mu_d = np.mean(dtu_arr, axis=0)
        mu_k = np.mean(kul_arr, axis=0)
        
        cos_dist = 1 - np.dot(mu_d, mu_k) / (np.linalg.norm(mu_d) * np.linalg.norm(mu_k) + 1e-12)
        
        sig_d = np.cov(dtu_arr, rowvar=False)
        sig_k = np.cov(kul_arr, rowvar=False)
        
        if sig_d.ndim == 0:
            sig_d, sig_k = np.array([[sig_d]]), np.array([[sig_k]])
            
        fd = calculate_frechet_distance(mu_d, sig_d, mu_k, sig_k)
        
        print(f"{layer:<20} | {dtu_arr.shape[1]:<5} | {cos_dist:<15.4f} | {fd:.4f}")

if __name__ == "__main__":
    run_adabn_experiment()
