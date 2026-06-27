import os
import sys
import numpy as np
import scipy.io
import scipy.signal
import torch
import math
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.experiment_10_layer_profiler import extract_gammatone_envelopes, subject_files, get_dtu_mapping_and_envelopes
from baselines.ridge_aad import load_subject_examples

def normalize_array(arr):
    arr = arr - arr.mean(axis=0, keepdims=True)
    scale = arr.std(axis=0, keepdims=True) + 1e-12
    return arr / scale

def get_dtu_tensor():
    s_files = subject_files()
    if not s_files:
        print("Missing DTU files.")
        return None, None, None
        
    s_file = s_files[0]
    subj = s_file.stem.split('_')[0]
    trials = load_subject_examples(s_file)
    mapping, envelopes = get_dtu_mapping_and_envelopes()
    dtu_indices = [13, 46, 43, 23, 50, 0, 52, 14]
    
    e_list, a_list, u_list = [], [], []
    win_len = int(3 * 64)
    
    for trial in trials:
        eeg_data = trial.eeg
        eeg_8 = eeg_data[:, dtu_indices]
        eeg_norm = normalize_array(eeg_8)
        
        trial_key = f"trial_{trial.trial_index}"
        if mapping and subj in mapping and trial_key in mapping[subj]:
            fname_a = mapping[subj][trial_key]["wavA"]["filename"]
            fname_b = mapping[subj][trial_key]["wavB"]["filename"]
            
            if fname_a in envelopes and fname_b in envelopes:
                env_a = envelopes[fname_a]
                env_b = envelopes[fname_b]
                
                env_att = env_a if trial.label == 1 else env_b
                env_unatt = env_b if trial.label == 1 else env_a
                
                env_att = normalize_array(env_att.T).T
                env_unatt = normalize_array(env_unatt.T).T
                
                min_len = min(len(eeg_norm), env_att.shape[1], env_unatt.shape[1])
                for start in range(0, min_len - win_len + 1, win_len):
                    e_list.append(eeg_norm[start:start+win_len].T)
                    a_list.append(env_att[:, start:start+win_len])
                    u_list.append(env_unatt[:, start:start+win_len])
                    
    # Return up to 100 windows
    return np.array(e_list)[:100], np.array(a_list)[:100], np.array(u_list)[:100]

def get_kul_tensor():
    mat_path = "/kaggle/input/datasets/lowk1ee/s1-klu/S1_KLU.mat"
    if not os.path.exists(mat_path):
        mat_path = "data/S1_KLU.mat"
    if not os.path.exists(mat_path):
        print("Missing KUL data.")
        return None, None, None
        
    mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    trials = mat['trials'] if 'trials' in mat else mat['trial']
    e_list, a_list, u_list = [], [], []
    win_len = int(3 * 64)
    
    for trial in trials:
        eeg_data = trial.RawData.EegData
        fs_eeg = trial.FileHeader.SampleRate
        channel_names = [ch.Label for ch in trial.FileHeader.Channels]
        
        target_channels = ['T7', 'C2', 'FT8', 'P7', 'CPz', 'Fp1', 'TP8', 'C3']
        
        # Recover 64-channel CAR from Cz-referenced KUL data
        eeg_data = eeg_data - eeg_data.mean(axis=1, keepdims=True)
        
        try:
            sel_idx = [channel_names.index(tc) if tc in channel_names else [c.upper() for c in channel_names].index(tc.upper()) for tc in target_channels]
        except ValueError as e:
            raise RuntimeError(f"Missing EEG channel in KUL dataset! {e}")
            
        eeg_8 = eeg_data[:, sel_idx]
        
        nyq = 0.5 * fs_eeg
        b, a = scipy.signal.butter(4, [1.0/nyq, 8.0/nyq], btype='band')
        eeg_8 = scipy.signal.filtfilt(b, a, eeg_8, axis=0)
        
        g = math.gcd(64, int(fs_eeg))
        eeg_8 = scipy.signal.resample_poly(eeg_8, 64 // g, int(fs_eeg) // g, axis=0)
        eeg_norm = normalize_array(eeg_8)
        
        att_ear = trial.attended_ear
        stimuli = trial.stimuli
        if len(stimuli) < 2: continue
        att_wav_name = str(stimuli[0] if att_ear == 'L' else stimuli[1])
        unatt_wav_name = str(stimuli[1] if att_ear == 'L' else stimuli[0])
        
        def find_wav(name):
            wav_dir = "/kaggle/input/datasets/lowk1ee/audio-klu" if os.path.exists("/kaggle/input/datasets/lowk1ee/audio-klu") else "data"
            for r, d, f in os.walk(wav_dir):
                for file in f:
                    if file == name or file == name + ".wav":
                        return os.path.join(r, file)
            return None
            
        att_wav_path = find_wav(att_wav_name)
        unatt_wav_path = find_wav(unatt_wav_name)
        
        if not att_wav_path or not unatt_wav_path:
            continue
            
        env_att = extract_gammatone_envelopes(att_wav_path, target_fs=64)
        env_unatt = extract_gammatone_envelopes(unatt_wav_path, target_fs=64)
        
        env_att = normalize_array(env_att.T).T
        env_unatt = normalize_array(env_unatt.T).T
        
        min_len = min(len(eeg_norm), env_att.shape[1], env_unatt.shape[1])
        for start in range(0, min_len - win_len + 1, win_len):
            e_list.append(eeg_norm[start:start+win_len].T)
            a_list.append(env_att[:, start:start+win_len])
            u_list.append(env_unatt[:, start:start+win_len])
            
    # Return up to 100 windows
    return np.array(e_list)[:100], np.array(a_list)[:100], np.array(u_list)[:100]

def print_stats(name, tensor):
    print(f"\n[{name}] Shape: {tensor.shape}")
    print(f"Mean: {tensor.mean():.6f}")
    print(f"Std:  {tensor.std():.6f}")
    print(f"Min:  {tensor.min():.6f}, Max: {tensor.max():.6f}")
    
    perc = np.percentile(tensor, [1, 5, 25, 50, 75, 95, 99])
    print("Percentiles [1, 5, 25, 50, 75, 95, 99]:")
    print(" ".join([f"{p:8.4f}" for p in perc]))

if __name__ == "__main__":
    print("="*60)
    print("PHASE G: INPUT EQUIVALENCE VERIFICATION")
    print("="*60)
    
    e_dtu, a_dtu, u_dtu = get_dtu_tensor()
    e_kul, a_kul, u_kul = get_kul_tensor()
    
    if e_kul is not None:
        assert e_dtu.shape == e_kul.shape, f"EEG shape mismatch: {e_dtu.shape} vs {e_kul.shape}"
        assert a_dtu.shape == a_kul.shape, f"Audio shape mismatch: {a_dtu.shape} vs {a_kul.shape}"
        assert u_dtu.shape == u_kul.shape, f"Unatt Audio shape mismatch: {u_dtu.shape} vs {u_kul.shape}"

        print("\n" + "="*30 + " EEG TENSOR " + "="*30)
        print_stats("DTU EEG", e_dtu)
        print_stats("KUL EEG", e_kul)
        
        print("\n" + "="*30 + " AUDIO TENSOR " + "="*30)
        print_stats("DTU AUDIO (Att)", a_dtu)
        print_stats("KUL AUDIO (Att)", a_kul)
        
        print("\n" + "="*30 + " CROSS-DATASET CORRELATION " + "="*30)
        print(f"EEG Pearson Correlation (flattened): {np.corrcoef(e_dtu.flatten(), e_kul.flatten())[0,1]:.6f}")
        print(f"Audio Pearson Correlation (flattened): {np.corrcoef(a_dtu.flatten(), a_kul.flatten())[0,1]:.6f}")
    else:
        print("Failed to load KUL tensors.")
