import os
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from scipy import signal
from scipy.io import wavfile as wav
from scipy.signal import welch
import time
import concurrent.futures
import multiprocessing as mp
from sklearn.model_selection import KFold

# -------------------------------------------------------------------------
# CONSTANTS & CONFIGURATION
# -------------------------------------------------------------------------
SR = 128
EAR_CHANNEL_INDICES = [23, 31, 32, 40, 14, 22, 41, 49]
MAX_LAG_SEC = 0.400 
MAX_LAG_SAMPLES = int(MAX_LAG_SEC * SR)

BROADBAND = (0.5, 8.0)
RIDGE_LAMBDA_GLOBAL = 100.0
PCA_COMPONENTS = 60

PRE_SWITCH_SAMPLES = int(0.5 * SR)
POST_SWITCH_SAMPLES = int(1.0 * SR)

MIN_SEGMENT_SEC = 3.0
MIN_SEGMENT_SAMPLES = int(MIN_SEGMENT_SEC * SR)

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

def get_masks(sp, length):
    mask_true = np.zeros(length, dtype=np.float32)
    mask_valid = np.ones(length, dtype=bool)
    
    if len(sp) == 0:
        mask_true[:] = 1.0
        return mask_true, mask_valid
        
    current_state = 1.0 if sp[0][0] == 'R' else 0.0 
    last_idx = 0
    for spk, idx in sp:
        end_idx = min(idx, length)
        mask_true[last_idx:end_idx] = current_state
        current_state = 1.0 if spk == 'L' else 0.0
        last_idx = end_idx
        
        b_start = max(0, idx - PRE_SWITCH_SAMPLES)
        b_end = min(length, idx + POST_SWITCH_SAMPLES)
        mask_valid[b_start:b_end] = False
        
        if last_idx >= length:
            break
            
    if last_idx < length:
        mask_true[last_idx:] = current_state
        
    return mask_true, mask_valid

def batch_pearsonr_pt(x, y):
    x_mean = x - x.mean(dim=0, keepdim=True)
    y_mean = y - y.mean(dim=0, keepdim=True)
    num = (x_mean * y_mean).sum(dim=0)
    den = torch.sqrt((x_mean**2).sum(dim=0) * (y_mean**2).sum(dim=0))
    return num / (den + 1e-8)

def solve_ridge_pt(XTX, XTy, lam=10.0):
    F = XTX.shape[0]
    I = torch.eye(F, device=XTX.device, dtype=XTX.dtype)
    jitter = 1e-6 * torch.randn(F, F, device=XTX.device, dtype=XTX.dtype) * I
    return torch.linalg.solve(XTX + lam * I + jitter, XTy)

def extract_segments(mask_valid):
    padded = np.concatenate([[False], mask_valid, [False]])
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts, ends))

def get_male_assignments():
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
            assignments[i] = 'L' 
    return assignments

def process_subject(cache_file, device_id, male_assignments):
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    subj_name = cache_file.stem.split('_')[0]
    
    cached = torch.load(cache_file, map_location='cpu', weights_only=False)['raw']
    N_TRIALS = len(cached)
    
    male_segments = []
    female_segments = []
    
    for tr_idx, tr in enumerate(cached):
        # 1-indexed for audio
        audio_id = tr_idx + 1
        male_ear = male_assignments.get(audio_id, 'L')
        
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
        
        X_trial = create_toeplitz_features_pt(eeg, MAX_LAG_SAMPLES)
        T_eff = X_trial.shape[0]
        
        mask_t, mask_v = get_masks(tr['meta'].get('switch_points', []), min_len)
        mask_t = torch.tensor(mask_t[MAX_LAG_SAMPLES:min_len], dtype=torch.float32, device=device)
        mask_v = mask_v[MAX_LAG_SAMPLES:min_len]
        
        YL = env_l[:T_eff]
        YR = env_r[:T_eff]
        
        Y_male = YL if male_ear == 'L' else YR
        Y_female = YR if male_ear == 'L' else YL
        
        segments = extract_segments(mask_v)
        
        for start, end in segments:
            if (end - start) < MIN_SEGMENT_SAMPLES:
                continue
                
            X_seg = X_trial[start:end]
            Ym_seg = Y_male[start:end]
            Yf_seg = Y_female[start:end]
            
            label_L = mask_t[start].item() == 1.0
            label_ear = 'L' if label_L else 'R'
            is_male_attended = (label_ear == male_ear)
            
            seg_data = {
                'X': X_seg,
                'Y_m': Ym_seg,
                'Y_f': Yf_seg
            }
            
            if is_male_attended:
                male_segments.append(seg_data)
            else:
                female_segments.append(seg_data)

    def evaluate_model(segments, target_is_male):
        if len(segments) < 4:
            return 0.5
            
        kf = KFold(n_splits=4, shuffle=True, random_state=42)
        
        all_correct = 0
        total_tested = 0
        
        # Precompute PCA components per fold
        for train_idx, test_idx in kf.split(segments):
            train_X = torch.cat([segments[i]['X'] for i in train_idx], dim=0)
            if target_is_male:
                train_Y = torch.cat([segments[i]['Y_m'] for i in train_idx], dim=0)
            else:
                train_Y = torch.cat([segments[i]['Y_f'] for i in train_idx], dim=0)
                
            U, S, V = torch.pca_lowrank(train_X, q=PCA_COMPONENTS)
            train_X_pca = torch.matmul(train_X, V)
            
            XTX = train_X_pca.T @ train_X_pca
            XTy = (train_X_pca.T @ train_Y).unsqueeze(-1)
            W = solve_ridge_pt(XTX, XTy, lam=RIDGE_LAMBDA_GLOBAL)
            
            for i in test_idx:
                test_X = segments[i]['X']
                test_Y_m = segments[i]['Y_m']
                test_Y_f = segments[i]['Y_f']
                
                test_X_pca = torch.matmul(test_X, V)
                Y_hat = (test_X_pca @ W).squeeze(-1)
                
                # IMPORTANT: Use Absolute Pearson correlation to handle dipole inversions natively!
                r_m = abs(batch_pearsonr_pt(Y_hat, test_Y_m).item())
                r_f = abs(batch_pearsonr_pt(Y_hat, test_Y_f).item())
                
                if target_is_male:
                    correct = r_m > r_f
                else:
                    correct = r_f > r_m
                    
                if correct:
                    all_correct += 1
                total_tested += 1
                
        return all_correct / total_tested
        
    acc_male = evaluate_model(male_segments, target_is_male=True)
    acc_female = evaluate_model(female_segments, target_is_male=False)
    
    overall = (acc_male * len(male_segments) + acc_female * len(female_segments)) / (len(male_segments) + len(female_segments) + 1e-8)
    
    print(f"  [{subj_name}] MALE Acc: {acc_male*100:.1f}% ({len(male_segments)} segs) | FEMALE Acc: {acc_female*100:.1f}% ({len(female_segments)} segs) | OVERALL: {overall*100:.1f}%")
    
    return subj_name, acc_male, acc_female, overall

def main():
    mp.set_start_method('spawn', force=True)
    
    print("Pre-computing Male audio assignments...")
    male_assignments = get_male_assignments()
    
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
    num_workers = min(mp.cpu_count(), num_gpus if num_gpus > 0 else mp.cpu_count())
    
    print(f"\n=======================================================")
    print(f" PHASE 142: SPEAKER-SPECIFIC SANITY CHECK")
    print(f" CPUs detected: {mp.cpu_count()} | GPUs detected: {num_gpus}")
    print(f"=======================================================\n", flush=True)
    
    cache_files = sorted(list(cache_dir.glob('*_multiband.pt')))
    
    start_time = time.time()
    final_results = {}
    
    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for idx, cache_file in enumerate(cache_files):
            device_id = idx % num_gpus if num_gpus > 0 else 0
            futures.append(executor.submit(process_subject, cache_file, device_id, male_assignments))
            
        for future in concurrent.futures.as_completed(futures):
            subj_name, m_acc, f_acc, o_acc = future.result()
            final_results[subj_name] = {'M': m_acc, 'F': f_acc, 'O': o_acc}

    print(f"\nTotal Pipeline Execution Time: {time.time() - start_time:.2f}s")
    print("\n=======================================================")
    print(" PHASE 142 SPEAKER-SPECIFIC ACCURACY")
    print("=======================================================")
    print(f"{'Subject':<10} {'Male Acc':<15} {'Female Acc':<15} {'Overall Acc':<15}")
    
    sorted_results = sorted(final_results.items(), key=lambda x: int(x[0][1:]))
    for subj, res in sorted_results:
        print(f"{subj:<10} {res['M']*100:.1f}%           {res['F']*100:.1f}%           {res['O']*100:.1f}%")

if __name__ == '__main__':
    main()
