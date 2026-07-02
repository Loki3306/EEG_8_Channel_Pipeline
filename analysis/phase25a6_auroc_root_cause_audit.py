import torch
import numpy as np
import scipy.io
import scipy.signal as signal
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.aad_conformer import AADConformer

def norm_env(env):
    env = env - env.mean(axis=1, keepdims=True)
    env = env / (env.std(axis=1, keepdims=True) + 1e-12)
    return env

def safe_corr_np(x, y, eps=1e-8):
    x_mean = x.mean(axis=1, keepdims=True)
    y_mean = y.mean(axis=1, keepdims=True)
    num = np.sum((x - x_mean) * (y - y_mean), axis=1)
    den = np.sqrt(np.sum((x - x_mean)**2, axis=1) * np.sum((y - y_mean)**2, axis=1))
    return num / (den + eps)

def run_diagnostic_inference(eeg_batch, model, device):
    eeg_batch = torch.tensor(eeg_batch, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        out, features = model(eeg_batch, return_features=True)
        pred_envs = out.squeeze(1).cpu().numpy()
        
    return pred_envs, features['z_pool'].cpu().numpy()

def main():
    print("====================================================")
    print("PHASE 25A.6: AUROC ROOT CAUSE AUDIT")
    print("====================================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AADConformer(in_channels=8)
    ckpt_path = "/kaggle/input/datasets/lowkieee/eeg-aad-conformer-seed1-checkpoints/checkpoints/seed_1/model_S1.pt"
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location='cpu')
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    mat_path = "/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG/S18/S18.mat"
    if not os.path.exists(mat_path):
        print(f"[FAIL] Missing {mat_path}. Please run on Kaggle.")
        sys.exit(1)
        
    mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    eeg_var = [k for k in mat.keys() if not k.startswith('__')][0]
    eeg_data = mat[eeg_var].data
    
    trial_eeg_128 = eeg_data[:, :, 0] if eeg_data.ndim == 3 else eeg_data[:, :7680]
    
    # Filter
    nyq = 128 / 2
    b, a = signal.butter(4, [1.0/nyq, 8.0/nyq], btype='band')
    eeg_filt = signal.filtfilt(b, a, trial_eeg_128, axis=1)
    
    # 1. WRONG CHANNELS (Phase 25a)
    eeg_wrong = eeg_filt[:8, :]
    
    # 2. RIGHT CHANNELS (Phase 25a5)
    fallback_map = {'T7': 14, 'C2': 43, 'FT8': 39, 'P7': 22, 'CPz': 31, 'Fp1': 0, 'TP8': 47, 'C3': 12}
    sel_idx = [fallback_map[tc] for tc in ['T7', 'C2', 'FT8', 'P7', 'CPz', 'Fp1', 'TP8', 'C3']]
    eeg_right = eeg_filt[sel_idx, :]
    
    import math
    g = math.gcd(64, 128)
    eeg_wrong_64 = signal.resample_poly(eeg_wrong, 64 // g, 128 // g, axis=1)
    eeg_right_64 = signal.resample_poly(eeg_right, 64 // g, 128 // g, axis=1)
    
    eeg_wrong_norm = norm_env(eeg_wrong_64)
    eeg_right_norm = norm_env(eeg_right_64)
    
    # Extract windows
    win_len = int(2.0 * 64)
    stride = int(1.0 * 64)
    
    def extract_windows(eeg_norm_arr):
        wins = []
        for start in range(0, eeg_norm_arr.shape[1] - win_len, stride):
            w = eeg_norm_arr[:, start:start+win_len]
            w_norm = (w - w.mean(axis=1, keepdims=True)) / (w.std(axis=1, keepdims=True) + 1e-8)
            wins.append(w_norm)
        return np.stack(wins)
        
    batch_wrong = extract_windows(eeg_wrong_norm)
    batch_right = extract_windows(eeg_right_norm)
    
    print(f"[INFO] Batches extracted: {batch_wrong.shape}")
    
    print(f"\n--- WRONG CHANNELS (Phase 25a) ---")
    print(f"Mean: {batch_wrong.mean():.4f}, Std: {batch_wrong.std():.4f}")
    pred_wrong, emb_wrong = run_diagnostic_inference(batch_wrong, model, device)
    print(f"Pred Output Variance: {pred_wrong.var(axis=1).mean():.4f}")
    print(f"Embedding Norm: {np.linalg.norm(emb_wrong, axis=1).mean():.4f}")
    
    print(f"\n--- RIGHT CHANNELS (Phase 25a5) ---")
    print(f"Mean: {batch_right.mean():.4f}, Std: {batch_right.std():.4f}")
    pred_right, emb_right = run_diagnostic_inference(batch_right, model, device)
    print(f"Pred Output Variance: {pred_right.var(axis=1).mean():.4f}")
    print(f"Embedding Norm: {np.linalg.norm(emb_right, axis=1).mean():.4f}")
    
    # Load audio to check correlation
    cache_path = "/kaggle/working/results/phase25a5/audio_cache/19.npz"
    if os.path.exists(cache_path):
        audio_cache = np.load(cache_path)
        env_l = audio_cache['env_l']
        env_r = audio_cache['env_r']
        
        l_wins, r_wins = [], []
        for start in range(0, len(env_l) - win_len, stride):
            if start+win_len > eeg_wrong_norm.shape[1]: break
            w_l = env_l[start:start+win_len]
            w_r = env_r[start:start+win_len]
            w_l = (w_l - w_l.mean()) / (w_l.std() + 1e-8)
            w_r = (w_r - w_r.mean()) / (w_r.std() + 1e-8)
            l_wins.append(w_l)
            r_wins.append(w_r)
            
        l_batch = np.stack(l_wins)[:len(batch_wrong)]
        r_batch = np.stack(r_wins)[:len(batch_wrong)]
        
        corr_l_wrong = safe_corr_np(pred_wrong, l_batch)
        corr_r_wrong = safe_corr_np(pred_wrong, r_batch)
        
        corr_l_right = safe_corr_np(pred_right, l_batch)
        corr_r_right = safe_corr_np(pred_right, r_batch)
        
        print("\n--- CORRELATIONS (S18 Trial 1) ---")
        print(f"Wrong Channels (Phase 25a) L-Corr Mean: {corr_l_wrong.mean():.4f}, R-Corr Mean: {corr_r_wrong.mean():.4f}")
        print(f"Right Channels (Phase 25a5) L-Corr Mean: {corr_l_right.mean():.4f}, R-Corr Mean: {corr_r_right.mean():.4f}")
    else:
        print(f"[FAIL] Missing audio cache at {cache_path}. Please generate it using phase25a5_dataset_validation.py first.")

if __name__ == "__main__":
    main()
