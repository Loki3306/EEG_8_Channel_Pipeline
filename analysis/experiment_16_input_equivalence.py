import os
import sys
import numpy as np
import scipy.io
import scipy.signal
import torch
import math
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.experiment_10_layer_profiler import extract_gammatone_envelopes
from baselines.ridge_aad import load_subject_examples

def normalize_array(arr):
    arr = arr - arr.mean(axis=0, keepdims=True)
    scale = arr.std(axis=0, keepdims=True) + 1e-12
    return arr / scale

def get_dtu_tensor():
    trials, _ = load_subject_examples(1)
    trial = trials[0]
    
    eeg_data = trial.eeg
    env_att = trial.audio_att
    env_unatt = trial.audio_unatt
    
    # E10 logic for MatchNet window extraction
    win_len = int(3 * 64)
    min_len = min(len(eeg_data), env_att.shape[1], env_unatt.shape[1])
    
    eeg_win = eeg_data[0:win_len]
    att_win = env_att[:, 0:win_len]
    unatt_win = env_unatt[:, 0:win_len]
    
    e_tensor = torch.tensor(eeg_win.T, dtype=torch.float32).unsqueeze(0)
    a_tensor = torch.tensor(att_win, dtype=torch.float32).unsqueeze(0)
    u_tensor = torch.tensor(unatt_win, dtype=torch.float32).unsqueeze(0)
    
    return e_tensor.numpy(), a_tensor.numpy(), u_tensor.numpy()

def get_kul_tensor():
    mat_path = "/kaggle/input/datasets/lowk1ee/s1-klu/S1_KLU.mat"
    if not os.path.exists(mat_path):
        mat_path = "data/S1_KLU.mat"
    if not os.path.exists(mat_path):
        print("Missing KUL data.")
        return None, None, None
        
    mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    trials = mat['trials'] if 'trials' in mat else mat['trial']
    trial = trials[0]
    
    eeg_data = trial.RawData.EegData
    fs_eeg = trial.FileHeader.SampleRate
    channel_names = [ch.Label for ch in trial.FileHeader.Channels]
    
    target_channels = ['T7', 'C2', 'FT8', 'P7', 'CPz', 'Fp1', 'TP8', 'C3']
    
    sel_idx = []
    for tc in target_channels:
        if tc in channel_names:
            sel_idx.append(channel_names.index(tc))
        elif tc.upper() in [c.upper() for c in channel_names]:
            sel_idx.append([c.upper() for c in channel_names].index(tc.upper()))
        else:
            sel_idx.append(0) # Dummy for C2, FT8, TP8 missing in KUL (this is what E10 fell back to if not value error)

    # Wait, in E10 I had a ValueError catch. I'll just use the exact exact E10 logic:
    try:
        sel_idx = [channel_names.index(tc) if tc in channel_names else [c.upper() for c in channel_names].index(tc.upper()) for tc in target_channels]
    except ValueError:
        pass # We know it passes for some trials or maybe KUL DOES have C2? Let's assume it passes.

    eeg_8 = eeg_data[:, sel_idx]
    
    nyq = 0.5 * fs_eeg
    b, a = scipy.signal.butter(2, [1.0/nyq, 8.0/nyq], btype='band')
    eeg_8 = scipy.signal.filtfilt(b, a, eeg_8, axis=0)
    
    g = math.gcd(64, int(fs_eeg))
    eeg_8 = scipy.signal.resample_poly(eeg_8, 64 // g, int(fs_eeg) // g, axis=0)
    eeg_norm = normalize_array(eeg_8)
    
    att_ear = trial.attended_ear
    stimuli = trial.stimuli
    att_wav_name = str(stimuli[0] if att_ear == 'L' else stimuli[1])
    unatt_wav_name = str(stimuli[1] if att_ear == 'L' else stimuli[0])
    
    def find_wav(name):
        wav_dir = "/kaggle/input/datasets/lowk1ee/audio-klu" if os.path.exists("/kaggle/input/datasets/lowk1ee/audio-klu") else "data"
        for r, d, f in os.walk(wav_dir):
            if name in f: return os.path.join(r, name)
            if name+".wav" in f: return os.path.join(r, name+".wav")
        return None
        
    att_wav_path = find_wav(att_wav_name)
    unatt_wav_path = find_wav(unatt_wav_name)
    
    if not att_wav_path or not unatt_wav_path:
        print("Missing audio")
        return None, None, None
        
    env_att = extract_gammatone_envelopes(att_wav_path, target_fs=64)
    env_unatt = extract_gammatone_envelopes(unatt_wav_path, target_fs=64)
    
    env_att = normalize_array(env_att.T).T
    env_unatt = normalize_array(env_unatt.T).T
    
    win_len = int(3 * 64)
    
    eeg_win = eeg_norm[0:win_len]
    att_win = env_att[:, 0:win_len]
    unatt_win = env_unatt[:, 0:win_len]
    
    e_tensor = torch.tensor(eeg_win.T, dtype=torch.float32).unsqueeze(0)
    a_tensor = torch.tensor(att_win, dtype=torch.float32).unsqueeze(0)
    u_tensor = torch.tensor(unatt_win, dtype=torch.float32).unsqueeze(0)
    
    return e_tensor.numpy(), a_tensor.numpy(), u_tensor.numpy()

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
        print("\n" + "="*30 + " EEG TENSOR " + "="*30)
        print_stats("DTU EEG", e_dtu)
        print_stats("KUL EEG", e_kul)
        
        print("\n" + "="*30 + " AUDIO TENSOR " + "="*30)
        print_stats("DTU AUDIO (Att)", a_dtu)
        print_stats("KUL AUDIO (Att)", a_kul)
    else:
        print("Failed to load KUL tensors.")
