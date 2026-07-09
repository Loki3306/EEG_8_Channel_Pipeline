import numpy as np
import torch
from pathlib import Path
from scipy import signal
import time
import concurrent.futures
import multiprocessing as mp
import scipy.io.wavfile as wav
from scipy.signal import welch

# -------------------------------------------------------------------------
# CONSTANTS & CONFIGURATION
# -------------------------------------------------------------------------
SR = 128
EAR_CHANNEL_INDICES = [23, 31, 32, 40, 14, 22, 41, 49]
MAX_LAG_SEC = 0.400 
MAX_LAG_SAMPLES = int(MAX_LAG_SEC * SR)
SEQ_SAMPLES = int(3.5 * SR)
SEQ_HOP = int(0.5 * SR)

BROADBAND = (0.5, 8.0)
RIDGE_LAMBDA = 100.0

def apply_modulation_filter(env, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, env, axis=1)

def create_toeplitz_features_pt(eeg, max_lag_samples):
    C, T = eeg.shape
    T_eff = T - max_lag_samples
    X = torch.zeros((T_eff, C * max_lag_samples), dtype=eeg.dtype, device=eeg.device)
    for tau in range(max_lag_samples):
        X[:, tau*C : (tau+1)*C] = eeg[:, tau : tau+T_eff].T
    return X

def batch_pearsonr_pt(x, y):
    x_mean = x - x.mean(dim=1, keepdim=True)
    y_mean = y - y.mean(dim=1, keepdim=True)
    num = (x_mean * y_mean).sum(dim=1)
    den = torch.sqrt((x_mean**2).sum(dim=1) * (y_mean**2).sum(dim=1))
    return num / (den + 1e-8)

def solve_ridge_pt(XTX, XTy, lam=100.0):
    F = XTX.shape[0]
    I = torch.eye(F, device=XTX.device, dtype=XTX.dtype)
    jitter = 1e-6 * torch.randn(F, F, device=XTX.device, dtype=XTX.dtype) * I
    return torch.linalg.solve(XTX + lam * I + jitter, XTy)

def prepare_trial_data(tr, device):
    eeg_raw = tr['eeg'].numpy()[EAR_CHANNEL_INDICES, :] 
    env_l_raw = tr['env_l'].numpy()
    env_r_raw = tr['env_r'].numpy()
    
    min_len = min(eeg_raw.shape[1], env_l_raw.shape[1])
    eeg_raw = eeg_raw[:, :min_len]
    env_l_raw = env_l_raw[:, :min_len]
    env_r_raw = env_r_raw[:, :min_len]
    
    eeg_f = apply_modulation_filter(eeg_raw, BROADBAND[0], BROADBAND[1], SR)
    env_l_f = apply_modulation_filter(env_l_raw, BROADBAND[0], BROADBAND[1], SR)
    env_r_f = apply_modulation_filter(env_r_raw, BROADBAND[0], BROADBAND[1], SR)
    
    eeg_f = (eeg_f - np.mean(eeg_f, axis=1, keepdims=True)) / (np.std(eeg_f, axis=1, keepdims=True) + 1e-8)
    env_l_f = (env_l_f - np.mean(env_l_f, axis=1, keepdims=True)) / (np.std(env_l_f, axis=1, keepdims=True) + 1e-8)
    env_r_f = (env_r_f - np.mean(env_r_f, axis=1, keepdims=True)) / (np.std(env_r_f, axis=1, keepdims=True) + 1e-8)
    
    eeg = torch.tensor(eeg_f, dtype=torch.float32, device=device)
    env_l = torch.tensor(env_l_f[0], dtype=torch.float32, device=device)
    env_r = torch.tensor(env_r_f[0], dtype=torch.float32, device=device)
    
    T = eeg.shape[1]
    X_trial = create_toeplitz_features_pt(eeg, MAX_LAG_SAMPLES)
    T_eff = X_trial.shape[0]
    F = X_trial.shape[1]
    
    Y_l_eff = env_l[:T_eff]
    Y_r_eff = env_r[:T_eff]
    
    sp = tr['meta']['switch_points']
    boundaries = [0] + [idx for spk, idx in sp]
    boundaries = sorted(set(boundaries))
    if boundaries[-1] != T: boundaries.append(T)
        
    mask_L = torch.zeros(T_eff, dtype=torch.bool, device=device)
    mask_R = torch.zeros(T_eff, dtype=torch.bool, device=device)
    
    seq_indices, Y_meta = [], []
    
    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i+1]
        
        current_spk = 'L'
        for spk, idx in sp:
            if idx <= start_idx: current_spk = spk
            else: break
            
        safe_eff_end = min(end_idx, T_eff)
        if start_idx < safe_eff_end:
            if current_spk == 'L':
                mask_L[start_idx:safe_eff_end] = True
            else:
                mask_R[start_idx:safe_eff_end] = True
            
        safe_start = start_idx + int(1.5 * SR)
        safe_end = end_idx
        
        if safe_end - safe_start >= SEQ_SAMPLES:
            for seq_start in range(safe_start, safe_end - SEQ_SAMPLES + 1, SEQ_HOP):
                if seq_start + SEQ_SAMPLES <= T_eff:
                    seq_indices.append((seq_start, seq_start + SEQ_SAMPLES))
                    Y_meta.append(1.0 if current_spk == 'L' else 0.0)
                
    X_L = X_trial[mask_L]
    Y_L = Y_l_eff[mask_L]
    
    X_R = X_trial[mask_R]
    Y_R = Y_r_eff[mask_R]
    
    XTX_L = (X_L.T @ X_L) if X_L.shape[0] > 0 else torch.zeros((F, F), device=device)
    XTy_L = (X_L.T @ Y_L).unsqueeze(-1) if X_L.shape[0] > 0 else torch.zeros((F, 1), device=device)
    
    XTX_R = (X_R.T @ X_R) if X_R.shape[0] > 0 else torch.zeros((F, F), device=device)
    XTy_R = (X_R.T @ Y_R).unsqueeze(-1) if X_R.shape[0] > 0 else torch.zeros((F, 1), device=device)
    
    return {
        'X': X_trial,
        'Y_l': Y_l_eff,
        'Y_r': Y_r_eff,
        'XTX_L': XTX_L,
        'XTy_L': XTy_L,
        'XTX_R': XTX_R,
        'XTy_R': XTy_R,
        'seq_indices': seq_indices,
        'Y_meta': torch.tensor(Y_meta, dtype=torch.float32, device=device)
    }

def evaluate_trial_scores(W_L, W_R, tr_info):
    if len(tr_info['seq_indices']) == 0:
        return None, None
        
    Y_hat_full_L = tr_info['X'] @ W_L.squeeze(-1) # (T_eff,)
    Y_hat_full_R = tr_info['X'] @ W_R.squeeze(-1) # (T_eff,)
    
    Y_hat_seqs_L = torch.stack([Y_hat_full_L[s:e] for s, e in tr_info['seq_indices']])
    Y_hat_seqs_R = torch.stack([Y_hat_full_R[s:e] for s, e in tr_info['seq_indices']])
    
    Y_l_seqs = torch.stack([tr_info['Y_l'][s:e] for s, e in tr_info['seq_indices']])
    Y_r_seqs = torch.stack([tr_info['Y_r'][s:e] for s, e in tr_info['seq_indices']])
    
    score_L = batch_pearsonr_pt(Y_hat_seqs_L, Y_l_seqs)
    score_R = batch_pearsonr_pt(Y_hat_seqs_R, Y_r_seqs)
    
    return score_L, score_R

def get_male_assignments():
    # Cache the male assignment for all 60 trials
    wav_dir = '/kaggle/input/datasets/lokeshgile/aasd-audio/Stimuli Audio'
    if not os.path.exists(wav_dir):
        wav_dir = '/kaggle/input/aasd-audio/Stimuli Audio'
        
    assignments = {}
    for i in range(1, 61):
        wav_path = os.path.join(wav_dir, f"mixed_{i:03d}.wav")
        if os.path.exists(wav_path):
            sr, data = wav.read(wav_path)
            f_L, Pxx_L = welch(data[:, 0], sr, nperseg=sr)
            f_R, Pxx_R = welch(data[:, 1], sr, nperseg=sr)
            valid_idx = (f_L >= 50) & (f_L <= 300)
            peak_L = f_L[valid_idx][np.argmax(Pxx_L[valid_idx])]
            peak_R = f_R[valid_idx][np.argmax(Pxx_R[valid_idx])]
            assignments[i] = 'L' if peak_L < peak_R else 'R'
        else:
            assignments[i] = 'L' # fallback
    return assignments

def process_subject(cache_file, device_id, male_assignments):
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    subj_name = cache_file.stem.split('_')[0]
    
    cached = torch.load(cache_file, map_location='cpu', weights_only=False)['raw']
    
    N_TRIALS = len(cached)
    data_cache = []
    
    for t_idx in range(N_TRIALS):
        data_cache.append(prepare_trial_data(cached[t_idx], device))
            
    scores_male_att = []
    scores_male_unatt = []
    scores_female_att = []
    scores_female_unatt = []
    
    for test_idx in range(N_TRIALS):
        train_indices = [i for i in range(N_TRIALS) if i != test_idx]
        
        XTX_train_L = sum([data_cache[i]['XTX_L'] for i in train_indices])
        XTy_train_L = sum([data_cache[i]['XTy_L'] for i in train_indices])
        W_final_L = solve_ridge_pt(XTX_train_L, XTy_train_L, lam=RIDGE_LAMBDA)
        
        XTX_train_R = sum([data_cache[i]['XTX_R'] for i in train_indices])
        XTy_train_R = sum([data_cache[i]['XTy_R'] for i in train_indices])
        W_final_R = solve_ridge_pt(XTX_train_R, XTy_train_R, lam=RIDGE_LAMBDA)
        
        tr_info = data_cache[test_idx]
        score_L, score_R = evaluate_trial_scores(W_final_L, W_final_R, tr_info)
        
        if score_L is not None:
            # Determine which is male and female for this trial
            male_ch = male_assignments[test_idx + 1] # 1-indexed
            
            score_L = score_L.cpu().numpy()
            score_R = score_R.cpu().numpy()
            y_meta = tr_info['Y_meta'].cpu().numpy()
            
            for i in range(len(score_L)):
                attended_L = (y_meta[i] == 1.0)
                
                if male_ch == 'L':
                    # L is Male, R is Female
                    if attended_L:
                        scores_male_att.append(score_L[i])
                        scores_female_unatt.append(score_R[i])
                    else:
                        scores_female_att.append(score_R[i])
                        scores_male_unatt.append(score_L[i])
                else:
                    # R is Male, L is Female
                    if attended_L:
                        scores_female_att.append(score_L[i])
                        scores_male_unatt.append(score_R[i])
                    else:
                        scores_male_att.append(score_R[i])
                        scores_female_unatt.append(score_L[i])
                        
    m_att = np.mean(scores_male_att) if scores_male_att else 0
    m_unatt = np.mean(scores_male_unatt) if scores_male_unatt else 0
    f_att = np.mean(scores_female_att) if scores_female_att else 0
    f_unatt = np.mean(scores_female_unatt) if scores_female_unatt else 0
    
    print(f"[{subj_name}] Male(Att): {m_att:.4f}, Male(Unatt): {m_unatt:.4f} | Female(Att): {f_att:.4f}, Female(Unatt): {f_unatt:.4f}")
    
    return subj_name

def main():
    mp.set_start_method('spawn', force=True)
    
    cache_dir = Path('/kaggle/working/multiband_cache')
    possible_paths = [
        Path('/kaggle/input/datasets/lowkieee/multiband-cache/kaggle/working/multiband_cache'),
        Path('/kaggle/input/multiband-cache/kaggle/working/multiband_cache'),
        Path('/kaggle/input/aasd-universal-cache-v1/kaggle/working/multiband_cache'),
        cache_dir
    ]
    for p in possible_paths:
        if p.exists() and len(list(p.glob('*_multiband.pt'))) > 0:
            cache_dir = p
            break
            
    num_gpus = torch.cuda.device_count()
    num_workers = mp.cpu_count()
    
    print("Pre-computing Male audio assignments...")
    male_assignments = get_male_assignments()
    
    print(f"\n=======================================================")
    print(f" PHASE 124: GPT SCORE CALIBRATION AUDIT")
    print(f" CPUs detected: {num_workers} | GPUs detected: {num_gpus}")
    print(f"=======================================================\n", flush=True)
    
    cache_files = sorted(list(cache_dir.glob('*_multiband.pt')))
    
    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(num_workers, len(cache_files))) as executor:
        for idx, cache_file in enumerate(cache_files):
            device_id = idx % num_gpus if num_gpus > 0 else 0
            futures.append(executor.submit(process_subject, cache_file, device_id, male_assignments))
            
        for future in concurrent.futures.as_completed(futures):
            _ = future.result()

if __name__ == '__main__':
    main()
