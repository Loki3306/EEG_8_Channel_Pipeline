import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.matchnet import ContrastiveMatchNet
from analysis.experiment_21_cross_evaluation import find_model

def run_filter_comparison():
    dtu_model_path = find_model(["matchnet", "fold"]) or find_model(["matchnet"])
    kul_model_path = find_model(["matchnet_kul_native"])
    
    if not dtu_model_path or not kul_model_path:
        print("Missing models. Need both DTU and KUL native checkpoints.")
        return
        
    model_dtu = ContrastiveMatchNet("eegnet")
    model_dtu.load_state_dict(torch.load(dtu_model_path, map_location='cpu'))
    
    model_kul = ContrastiveMatchNet("eegnet")
    model_kul.load_state_dict(torch.load(kul_model_path, map_location='cpu'))
    
    dtu_spatial = model_dtu.eeg_encoder.block1[2].weight.detach().cpu().numpy().squeeze() # (16, 8)
    kul_spatial = model_kul.eeg_encoder.block1[2].weight.detach().cpu().numpy().squeeze() # (16, 8)
    
    channels = ['T7', 'C2', 'FT8', 'P7', 'CPz', 'Fp1', 'TP8', 'C3']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    vmin = min(np.min(dtu_spatial), np.min(kul_spatial))
    vmax = max(np.max(dtu_spatial), np.max(kul_spatial))
    vmax = max(abs(vmin), abs(vmax))
    vmin = -vmax
    
    im1 = axes[0].imshow(dtu_spatial, aspect='auto', cmap='coolwarm', vmin=vmin, vmax=vmax)
    axes[0].set_title("Optimal DTU Spatial Filters")
    axes[0].set_xticks(np.arange(8))
    axes[0].set_xticklabels(channels)
    axes[0].set_yticks(np.arange(16))
    axes[0].set_ylabel("Filter Index")
    
    im2 = axes[1].imshow(kul_spatial, aspect='auto', cmap='coolwarm', vmin=vmin, vmax=vmax)
    axes[1].set_title("Optimal KUL Spatial Filters (S1)")
    axes[1].set_xticks(np.arange(8))
    axes[1].set_xticklabels(channels)
    axes[1].set_yticks(np.arange(16))
    
    fig.colorbar(im1, ax=axes.ravel().tolist(), label="Weight")
    plt.savefig("optimal_filter_comparison.png")
    plt.close()
    
    print("Saved 'optimal_filter_comparison.png'.")

if __name__ == "__main__":
    run_filter_comparison()
