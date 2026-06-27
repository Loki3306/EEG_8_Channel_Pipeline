import os
import sys
import numpy as np
import scipy.io
import scipy.signal
from pathlib import Path
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baselines.ridge_aad import load_subject_examples

def butter_bandpass_filter(data, lowcut, highcut, fs, order=1, axis=0):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    return scipy.signal.filtfilt(b, a, data, axis=axis)

def normalize_array(arr):
    arr = arr - arr.mean(axis=0, keepdims=True)
    scale = arr.std(axis=0, keepdims=True) + 1e-12
    return arr / scale

def get_stats(data_name, arr, fs):
    # arr expected to be [T, Channels]
    means = np.mean(arr, axis=0)
    stds = np.std(arr, axis=0)
    rms = np.sqrt(np.mean(arr**2, axis=0))
    cov = np.cov(arr, rowvar=False)
    try:
        cond = np.linalg.cond(cov)
    except:
        cond = float('inf')
        
    # PSD Peak Frequency on first channel
    f, pxx = scipy.signal.welch(arr[:, 0], fs=fs, nperseg=fs*2)
    peak_f = f[np.argmax(pxx)]
    
    return {
        "Shape": arr.shape,
        "FS": fs,
        "Mean": f"{np.mean(means):.4e}",
        "Std": f"{np.mean(stds):.4e}",
        "Min": f"{np.min(arr):.4e}",
        "Max": f"{np.max(arr):.4e}",
        "RMS": f"{np.mean(rms):.4e}",
        "CondNum": f"{cond:.4e}",
        "PeakHz": f"{peak_f:.2f}"
    }

def print_stats_table(step_name, dtu_stats, kul_stats):
    print(f"\n{'-'*80}")
    print(f"STEP: {step_name}")
    print(f"{'-'*80}")
    
    keys = list(dtu_stats.keys())
    print(f"{'Metric':<15} | {'DTU':<25} | {'KUL':<25}")
    print("-" * 80)
    for k in keys:
        print(f"{k:<15} | {str(dtu_stats[k]):<25} | {str(kul_stats[k]):<25}")

def run_pipeline_audit():
    print("="*80)
    print("PHASE E: PREPROCESSING PIPELINE AUDIT")
    print("="*80)
    
    # 1. Load one trial of DTU
    dtu_path = Path("/kaggle/input/datasets/lokeshgile/dataset-eeg/S1_data_preproc.mat")
    if not dtu_path.exists():
        dtu_path = Path("data/S1_data_preproc.mat") # Local fallback
        
    if not dtu_path.exists():
        print("DTU data not found.")
        return
        
    print(f"Loading DTU from {dtu_path}...")
    dtu_examples = load_subject_examples(dtu_path)
    dtu_raw = dtu_examples[0].eeg # [T, 64]
    
    # 2. Load one trial of KUL
    kul_path = Path("/kaggle/input/datasets/lowk1ee/s1-klu/S1_KLU.mat")
    if not kul_path.exists():
        print("KUL data not found.")
        return
        
    print(f"Loading KUL from {kul_path}...")
    kul_mat = scipy.io.loadmat(kul_path)
    # Parse KUL trial 1
    kul_trials = []
    if 'trials' in kul_mat:
        for i in range(kul_mat['trials'].shape[1]):
            t = kul_mat['trials'][0, i]
            eeg = t['RawData'][0,0]
            kul_trials.append(eeg)
    kul_raw = kul_trials[0] # [T, 64]
    
    # --- PIPELINE STEP 1: RAW 8-CHANNEL SELECTION ---
    dtu_channels = [13, 46, 43, 23, 50, 0, 52, 14]
    
    kul_channel_names = [
        'Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3', 'Pz', 'PO3', 'O1', 'Oz',
        'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz',
        'EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8', 'GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp',
        'Plet', 'Temp', 'Status', 'Fp1_2', 'AF3_2', 'F7_2', 'F3_2', 'FC1_2', 'FC5_2', 'T7_2', 'C3_2', 'CP1_2',
        'CP5_2', 'P7_2', 'P3_2', 'Pz_2', 'PO3_2', 'O1_2', 'Oz_2'
    ]
    kul_target_channels = ['T7', 'C2', 'FT8', 'P7', 'CPz', 'Fp1', 'TP8', 'C3']
    
    # We map KUL target channels to indices. Some are missing, so we use the nearest approximation 
    # from our MatchNet logic in E10.
    kul_sel_idx = []
    for tc in kul_target_channels:
        if tc in kul_channel_names:
            kul_sel_idx.append(kul_channel_names.index(tc))
        elif tc.upper() in [c.upper() for c in kul_channel_names]:
            kul_sel_idx.append([c.upper() for c in kul_channel_names].index(tc.upper()))
        else:
            # Fallbacks used in E10:
            if tc == 'C2': kul_sel_idx.append(kul_channel_names.index('Cz'))
            elif tc == 'FT8': kul_sel_idx.append(kul_channel_names.index('F8'))
            elif tc == 'TP8': kul_sel_idx.append(kul_channel_names.index('T8'))
            else: kul_sel_idx.append(0)
            
    dtu_s1 = dtu_raw[:, dtu_channels]
    kul_s1 = kul_raw[:, kul_sel_idx]
    
    dtu_fs = 64
    kul_fs = 128
    
    print_stats_table("1. RAW SELECTION", get_stats("DTU", dtu_s1, dtu_fs), get_stats("KUL", kul_s1, kul_fs))
    
    # --- PIPELINE STEP 2: BANDPASS FILTERING (1-8 Hz) ---
    # In DTU, train_matchnet_loso does butter_bandpass_filter on axis=1 of transpose
    dtu_s2 = butter_bandpass_filter(dtu_s1.T, 1, 8, dtu_fs, axis=1).T
    
    # In KUL, E10 does butter_bandpass_filter on axis=0
    kul_s2 = butter_bandpass_filter(kul_s1, 1, 8, kul_fs, axis=0)
    
    print_stats_table("2. BANDPASS (1-8 Hz)", get_stats("DTU", dtu_s2, dtu_fs), get_stats("KUL", kul_s2, kul_fs))
    
    # --- PIPELINE STEP 3: RESAMPLING KUL (128 -> 64) ---
    dtu_s3 = dtu_s2
    g = math.gcd(dtu_fs, kul_fs)
    kul_s3 = scipy.signal.resample_poly(kul_s2, dtu_fs // g, kul_fs // g, axis=0)
    
    print_stats_table("3. RESAMPLING (64 Hz)", get_stats("DTU", dtu_s3, dtu_fs), get_stats("KUL", kul_s3, dtu_fs))
    
    # --- PIPELINE STEP 4: NORMALIZATION (Zero Mean, Unit Var) ---
    dtu_s4 = normalize_array(dtu_s3)
    kul_s4 = normalize_array(kul_s3)
    
    print_stats_table("4. Z-SCORE NORMALIZATION", get_stats("DTU", dtu_s4, dtu_fs), get_stats("KUL", kul_s4, dtu_fs))

if __name__ == "__main__":
    run_pipeline_audit()
