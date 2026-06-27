import os
import sys
import pickle
import numpy as np
import scipy.io
import scipy.signal
import torch
import math
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.experiment_10_layer_profiler import (
    load_matchnet, 
    calculate_frechet_distance, 
    LayerProfiler,
    extract_gammatone_envelopes
)
from baselines.ridge_aad import load_subject_examples

def butter_bandpass_filter(data, lowcut, highcut, fs, order=1, axis=0):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    return scipy.signal.filtfilt(b, a, data, axis=axis)

def butter_highpass_filter(data, cutoff, fs, order=2, axis=0):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype='high', analog=False)
    return scipy.signal.filtfilt(b, a, data, axis=axis)

def normalize_array(arr):
    arr = arr - arr.mean(axis=0, keepdims=True)
    scale = arr.std(axis=0, keepdims=True) + 1e-12
    return arr / scale

def get_dtu_mapping_and_envelopes():
    import json
    REPO_ROOT = Path(__file__).resolve().parents[1]
    
    kaggle_map_dir = Path("/kaggle/input/datasets/lokeshgile/dataset-eeg")
    if (kaggle_map_dir / "audio_mapping.json").exists():
        map_file = kaggle_map_dir / "audio_mapping.json"
    else:
        map_file = REPO_ROOT / "data" / "audio_mapping.json"
    
    kaggle_env_dir = Path("/kaggle/input/datasets/lokeshgile/gammatone-envelope")
    if kaggle_env_dir.exists() and list(kaggle_env_dir.glob("*.pkl")):
        env_file = list(kaggle_env_dir.glob("*.pkl"))[0]
    else:
        env_file = REPO_ROOT / "data" / "gammatone_envelopes.pkl"
        
    if not map_file.exists() or not env_file.exists():
        return None, None
        
    with open(map_file, 'r') as f:
        mapping = json.load(f)
    with open(env_file, 'rb') as f:
        envelopes = pickle.load(f)
    return mapping, envelopes

def run_emulation():
    print("="*80)
    print("PHASE F: DTU PREPROCESSING EMULATOR")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_matchnet(device)
    if model is None: return
    
    dtu_cache_path = 'data/DTU_Layer_Activations.pkl'
    if not os.path.exists(dtu_cache_path):
        print("Run E10 first to cache DTU activations.")
        return
    with open(dtu_cache_path, 'rb') as f:
        dtu_act = pickle.load(f)
        
    dtu_b1 = dtu_act["EEG_Block1"]
    mu_d = np.mean(dtu_b1, axis=0)
    sig_d = np.cov(dtu_b1, rowvar=False)

    mapping, envelopes = get_dtu_mapping_and_envelopes()
    if mapping is None:
        print("Missing audio mapping/envelopes.")
        return

    # Load KUL Data
    trials, _ = load_subject_examples(1)
    if not trials:
        print("KUL data missing.")
        return
    
    kul_channel_names = [
        'Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3', 'Pz', 'PO3', 'O1', 'Oz',
        'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz',
        'EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8', 'GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp',
        'Plet', 'Temp', 'Status', 'Fp1_2', 'AF3_2', 'F7_2', 'F3_2', 'FC1_2', 'FC5_2', 'T7_2', 'C3_2', 'CP1_2',
        'CP5_2', 'P7_2', 'P3_2', 'Pz_2', 'PO3_2', 'O1_2', 'Oz_2'
    ]
    kul_target_channels = ['T7', 'C2', 'FT8', 'P7', 'CPz', 'Fp1', 'TP8', 'C3']
    
    kul_sel_idx = []
    for tc in kul_target_channels:
        if tc in kul_channel_names:
            kul_sel_idx.append(kul_channel_names.index(tc))
        elif tc.upper() in [c.upper() for c in kul_channel_names]:
            kul_sel_idx.append([c.upper() for c in kul_channel_names].index(tc.upper()))
        else:
            if tc == 'C2': kul_sel_idx.append(kul_channel_names.index('Cz'))
            elif tc == 'FT8': kul_sel_idx.append(kul_channel_names.index('F8'))
            elif tc == 'TP8': kul_sel_idx.append(kul_channel_names.index('T8'))
            else: kul_sel_idx.append(0)
            
    cz_idx = kul_channel_names.index('Cz')

    # Define Preprocessing Schemes to test
    # Each function takes raw_eeg (T, 64) at 128 Hz and returns eeg_norm (T_new, 8) at 64 Hz
    def scheme_baseline(raw):
        eeg_8 = raw[:, kul_sel_idx]
        eeg_8 = butter_bandpass_filter(eeg_8, 1, 8, 128, axis=0)
        eeg_8 = scipy.signal.resample_poly(eeg_8, 1, 2, axis=0)
        return normalize_array(eeg_8)
        
    def scheme_cz_ref(raw):
        ref = raw[:, cz_idx:cz_idx+1]
        eeg_ref = raw - ref
        eeg_8 = eeg_ref[:, kul_sel_idx]
        eeg_8 = butter_bandpass_filter(eeg_8, 1, 8, 128, axis=0)
        eeg_8 = scipy.signal.resample_poly(eeg_8, 1, 2, axis=0)
        return normalize_array(eeg_8)
        
    def scheme_avg_ref(raw):
        # Exclude EXG and Status channels for average ref, standard practice
        eeg_chans = raw[:, :32] 
        ref = np.mean(eeg_chans, axis=1, keepdims=True)
        eeg_ref = raw - ref
        eeg_8 = eeg_ref[:, kul_sel_idx]
        eeg_8 = butter_bandpass_filter(eeg_8, 1, 8, 128, axis=0)
        eeg_8 = scipy.signal.resample_poly(eeg_8, 1, 2, axis=0)
        return normalize_array(eeg_8)

    def scheme_dtu_matlab_exact(raw):
        # Emulate `preproc_data (2).m` exactly
        # 1. Average reference (using 'all' non-EOG channels)
        eeg_chans = raw[:, :32]
        ref = np.mean(eeg_chans, axis=1, keepdims=True)
        eeg_ref = raw - ref
        
        # 2. Downsample to 64Hz
        eeg_64 = scipy.signal.resample_poly(eeg_ref, 1, 2, axis=0)
        
        # 3. 0.1Hz Highpass
        eeg_64 = butter_highpass_filter(eeg_64, 0.1, 64, order=2, axis=0)
        
        # 4. DTU Python pipeline: Extract 8 channels, 1-8Hz bandpass, normalize
        eeg_8 = eeg_64[:, kul_sel_idx]
        eeg_8 = butter_bandpass_filter(eeg_8.T, 1, 8, 64, axis=1).T
        return normalize_array(eeg_8)

    schemes = {
        "1_Baseline (No Ref)": scheme_baseline,
        "2_Cz_Reference": scheme_cz_ref,
        "3_Average_Reference": scheme_avg_ref,
        "4_DTU_MATLAB_Emulator": scheme_dtu_matlab_exact
    }
    
    print(f"{'Scheme':<25} | {'Cosine Dist':<15} | {'Fréchet Distance'}")
    print("-" * 70)
    
    for scheme_name, preproc_fn in schemes.items():
        profiler = LayerProfiler(model, device)
        profiler.clear()
        
        # Process first 5 trials to get a good statistical estimate without taking 6 minutes
        for i in range(min(5, len(trials))):
            trial = trials[i]
            raw_eeg = trial.RawData.EegData # [T, 64]
            att_ear = trial.attended_ear
            stimuli = trial.stimuli
            if len(stimuli) < 2:
                continue

            
            att_wav_name = str(stimuli[0] if att_ear == 'L' else stimuli[1])
            unatt_wav_name = str(stimuli[1] if att_ear == 'L' else stimuli[0])
            
            def find_wav(name):
                wav_dir = "/kaggle/input/datasets/lokeshgile/dataset-eeg" if os.path.exists("/kaggle/input/datasets/lokeshgile/dataset-eeg") else "data"
                for r, d, f in os.walk(wav_dir):
                    if name in f: return os.path.join(r, name)
                    if name+".wav" in f: return os.path.join(r, name+".wav")
                return None
                
            att_wav_path = find_wav(att_wav_name)
            unatt_wav_path = find_wav(unatt_wav_name)
            
            if not att_wav_path or not unatt_wav_path:
                print(f"DEBUG: Could not find wavs for {att_wav_name} or {unatt_wav_name}")
                continue
                
            if att_wav_path and unatt_wav_path:
                if 'audio_cache' not in locals():
                    audio_cache = {}
                if att_wav_name not in audio_cache:
                    audio_cache[att_wav_name] = extract_gammatone_envelopes(att_wav_path, target_fs=64)
                if unatt_wav_name not in audio_cache:
                    audio_cache[unatt_wav_name] = extract_gammatone_envelopes(unatt_wav_path, target_fs=64)
                    
                env_att = audio_cache[att_wav_name]
                env_unatt = audio_cache[unatt_wav_name]
                
                env_att = normalize_array(env_att.T).T
                env_unatt = normalize_array(env_unatt.T).T
                
                # Apply Preprocessing Scheme!
                eeg_norm = preproc_fn(raw_eeg)
                
                min_len = min(len(eeg_norm), env_att.shape[1], env_unatt.shape[1])
                win_len = int(3 * 64)
                stride = int(1.5 * 64)
                
                for start in range(0, min_len - win_len + 1, stride):
                    eeg_win = eeg_norm[start:start+win_len]
                    att_win = env_att[:, start:start+win_len]
                    unatt_win = env_unatt[:, start:start+win_len]
                    
                    e_tensor = torch.tensor(eeg_win.T, dtype=torch.float32).unsqueeze(0).to(device)
                    a_tensor = torch.tensor(att_win, dtype=torch.float32).unsqueeze(0).to(device)
                    u_tensor = torch.tensor(unatt_win, dtype=torch.float32).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        profiler.model(e_tensor, a_tensor, u_tensor)
                            
        acts = profiler.get_activations()
        if "EEG_Block1" not in acts or len(acts["EEG_Block1"]) == 0:
            print(f"{scheme_name:<25} | FAILED")
            continue
            
        kul_b1 = acts["EEG_Block1"]
        mu_k = np.mean(kul_b1, axis=0)
        sig_k = np.cov(kul_b1, rowvar=False)
        
        cos_dist = 1 - np.dot(mu_d, mu_k) / (np.linalg.norm(mu_d) * np.linalg.norm(mu_k) + 1e-12)
        fd = calculate_frechet_distance(mu_d, sig_d, mu_k, sig_k)
        
        print(f"{scheme_name:<25} | {cos_dist:<15.4f} | {fd:.4f}")

if __name__ == "__main__":
    run_emulation()
