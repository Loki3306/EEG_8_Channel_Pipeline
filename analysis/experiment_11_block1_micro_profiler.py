import os
import sys
import pickle
import numpy as np
import scipy.linalg
import torch
import torch.nn.functional as F

# Add repository root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.experiment_10_layer_profiler import (
    load_matchnet,
    compute_dtu_activations,
    compute_kul_activations,
    calculate_frechet_distance
)

class MicroProfiler:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.activations = {}
        
        self.hooks = []
        self._register_hooks()
        
    def _register_hooks(self):
        def get_hook(name):
            def hook(module, input, output):
                # We expect 4 dims [B, C, H, W] from Block1 Conv/BN layers
                if len(output.shape) == 4:
                    pooled = output.mean(dim=(2, 3)) # -> [B, C]
                elif len(output.shape) == 3:
                    pooled = output.mean(dim=2) # -> [B, C]
                else:
                    pooled = output
                    
                if name not in self.activations:
                    self.activations[name] = []
                self.activations[name].append(pooled.detach().cpu().numpy()[0])
            return hook
            
        # Hook inside EEGNet block1
        block1 = self.model.eeg_encoder.block1
        self.hooks.append(block1[0].register_forward_hook(get_hook("01_TemporalConv")))
        self.hooks.append(block1[1].register_forward_hook(get_hook("02_BatchNorm1")))
        self.hooks.append(block1[2].register_forward_hook(get_hook("03_SpatialConv")))
        self.hooks.append(block1[3].register_forward_hook(get_hook("04_BatchNorm2")))
        self.hooks.append(block1[4].register_forward_hook(get_hook("05_GELU")))
        
    def clear(self):
        self.activations = {k: [] for k in self.activations}
        
    def get_activations(self):
        return {k: np.array(v) for k, v in self.activations.items()}
        
    def remove_hooks(self):
        for h in self.hooks:
            h.remove()

def run_micro_profiling():
    print("="*70)
    print("PHASE D1: EEG_BLOCK1 MICRO-PROFILER")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_matchnet(device)
    if model is None:
        return
        
    profiler = MicroProfiler(model, device)
    
    os.makedirs('data', exist_ok=True)
    dtu_cache_path = 'data/DTU_Micro_Activations.pkl'
    if os.path.exists(dtu_cache_path):
        print(f"Loading DTU micro-activations from cache {dtu_cache_path}...")
        with open(dtu_cache_path, 'rb') as f:
            dtu_act = pickle.load(f)
    else:
        dtu_act = compute_dtu_activations(profiler)
        if dtu_act:
            with open(dtu_cache_path, 'wb') as f:
                pickle.dump(dtu_act, f)
                
    kul_act = compute_kul_activations(profiler)
    
    if not dtu_act or not kul_act:
        print("Missing dataset. Aborting.")
        return
        
    print("\n" + "="*70)
    print("MICRO-DIVERGENCE (Fréchet Distance & Cosine Distance)")
    print("="*70)
    
    layers = [
        "01_TemporalConv", "02_BatchNorm1", "03_SpatialConv",
        "04_BatchNorm2", "05_GELU"
    ]
    
    print(f"{'Operation':<20} | {'Dim':<5} | {'Cosine Dist':<15} | {'Fréchet Distance'}")
    print("-" * 70)
    
    for layer in layers:
        if layer not in dtu_act or layer not in kul_act: continue
        
        dtu_arr = dtu_act[layer]
        kul_arr = kul_act[layer]
        
        if len(dtu_arr) == 0 or len(kul_arr) == 0: continue
        
        # Means
        mu_d = np.mean(dtu_arr, axis=0)
        mu_k = np.mean(kul_arr, axis=0)
        
        # Cosine distance of centroids
        cos_dist = 1 - np.dot(mu_d, mu_k) / (np.linalg.norm(mu_d) * np.linalg.norm(mu_k) + 1e-12)
        
        # Covariances
        sig_d = np.cov(dtu_arr, rowvar=False)
        sig_k = np.cov(kul_arr, rowvar=False)
        
        if sig_d.ndim == 0:
            sig_d = np.array([[sig_d]])
            sig_k = np.array([[sig_k]])
            
        fd = calculate_frechet_distance(mu_d, sig_d, mu_k, sig_k)
        
        print(f"{layer:<20} | {dtu_arr.shape[1]:<5} | {cos_dist:<15.4f} | {fd:.4f}")

if __name__ == "__main__":
    run_micro_profiling()
