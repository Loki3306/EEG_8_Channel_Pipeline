import os
import sys
import numpy as np
import scipy.signal
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.experiment_16_input_equivalence import get_dtu_tensor, get_kul_tensor
from models.matchnet import MatchNet

def compute_correlation_matrix(eeg_tensor):
    # eeg_tensor shape: (N, C, T) -> (100, 8, 192)
    # We want to compute an 8x8 correlation matrix across all windows and timepoints.
    # Flatten N and T: shape becomes (C, N*T)
    C = eeg_tensor.shape[1]
    flattened = eeg_tensor.transpose(1, 0, 2).reshape(C, -1)
    return np.corrcoef(flattened)

def compute_psd(eeg_tensor, fs=64):
    # Compute PSD for each channel using Welch's method
    # eeg_tensor shape: (N, C, T)
    N, C, T = eeg_tensor.shape
    psds = []
    freqs = None
    for c in range(C):
        # Flatten all windows for this channel to get a single continuous-like signal
        f, Pxx = scipy.signal.welch(eeg_tensor[:, c, :].flatten(), fs=fs, nperseg=T)
        psds.append(Pxx)
        freqs = f
    return freqs, np.array(psds)

def compute_band_power(freqs, psds):
    # bands: delta (1-4), theta (4-8), alpha (8-13), beta (13-30)
    bands = {
        'Delta (1-4 Hz)': (1, 4),
        'Theta (4-8 Hz)': (4, 8),
        'Alpha (8-13 Hz)': (8, 13),
        'Beta (13-30 Hz)': (13, 30)
    }
    
    band_powers = {b: [] for b in bands}
    for c in range(psds.shape[0]):
        for b_name, (low, high) in bands.items():
            idx = np.logical_and(freqs >= low, freqs <= high)
            band_power = np.trapz(psds[c, idx], freqs[idx])
            band_powers[b_name].append(band_power)
            
    return band_powers

def print_matrix(mat, title):
    print(f"\n--- {title} ---")
    np.set_printoptions(precision=3, suppress=True, linewidth=120)
    print(mat)

def run_experiment():
    print("Loading DTU tensors...")
    e_dtu, _, _ = get_dtu_tensor()
    print("Loading KUL tensors...")
    e_kul, _, _ = get_kul_tensor()
    
    if e_kul is None or e_dtu is None:
        print("Failed to load tensors.")
        return
        
    print(f"Loaded DTU shape: {e_dtu.shape}")
    print(f"Loaded KUL shape: {e_kul.shape}")
    
    # 1. Spatial Correlation
    corr_dtu = compute_correlation_matrix(e_dtu)
    corr_kul = compute_correlation_matrix(e_kul)
    
    print_matrix(corr_dtu, "DTU 8x8 Spatial Correlation Matrix")
    print_matrix(corr_kul, "KUL 8x8 Spatial Correlation Matrix")
    
    # Diff matrix to see what changed most
    print_matrix(np.abs(corr_dtu - corr_kul), "Absolute Difference Matrix (|DTU - KUL|)")
    
    # 2. Power Spectral Density
    freqs, psd_dtu = compute_psd(e_dtu)
    _, psd_kul = compute_psd(e_kul)
    
    bp_dtu = compute_band_power(freqs, psd_dtu)
    bp_kul = compute_band_power(freqs, psd_kul)
    
    print("\n--- Average Band Power across all channels ---")
    print(f"{'Band':<16} | {'DTU':<10} | {'KUL':<10} | {'Ratio (DTU/KUL)':<15}")
    print("-" * 60)
    for b in bp_dtu:
        mean_dtu = np.mean(bp_dtu[b])
        mean_kul = np.mean(bp_kul[b])
        ratio = mean_dtu / mean_kul if mean_kul > 0 else 0
        print(f"{b:<16} | {mean_dtu:<10.4f} | {mean_kul:<10.4f} | {ratio:<15.2f}")
        
    # 3. Spatial Filter Response
    print("\n--- MatchNet Spatial Filter Response ---")
    model_path = "/kaggle/input/datasets/lowk1ee/matchnet-checkpoints/best_model.pt"
    if not os.path.exists(model_path):
        model_path = "checkpoints/best_model.pt"
        
    if not os.path.exists(model_path):
        print("Could not find MatchNet checkpoint.")
        return
        
    model = MatchNet(fs=64)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Tensors are (N, C, T) -> need (N, 1, C, T) for MatchNet
    t_dtu = torch.tensor(e_dtu, dtype=torch.float32).unsqueeze(1)
    t_kul = torch.tensor(e_kul, dtype=torch.float32).unsqueeze(1)
    
    with torch.no_grad():
        # MatchNet EEG Branch:
        # Pass through temporal filters
        temp_dtu = model.eeg_branch[0:3](t_dtu)
        temp_kul = model.eeg_branch[0:3](t_kul)
        
        # Pass through spatial filters
        spat_dtu = model.eeg_branch[3](temp_dtu)
        spat_kul = model.eeg_branch[3](temp_kul)
        
        # Compute variance per spatial filter (dimension 1 is the 16 filters)
        var_dtu = spat_dtu.var(dim=(0, 2, 3)).numpy()
        var_kul = spat_kul.var(dim=(0, 2, 3)).numpy()
        
    print(f"{'Filter':<8} | {'DTU Variance':<15} | {'KUL Variance':<15} | {'Ratio (KUL/DTU)':<15}")
    print("-" * 65)
    for i in range(16):
        ratio = var_kul[i] / var_dtu[i] if var_dtu[i] > 0 else 0
        print(f"Filter {i:<1} | {var_dtu[i]:<15.4f} | {var_kul[i]:<15.4f} | {ratio:<15.2f}")
        
if __name__ == "__main__":
    run_experiment()
