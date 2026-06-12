import os
import sys
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scipy.signal import butter, filtfilt
from baselines.ridge_aad import (
    subject_files,
    load_subject_examples,
)

MAPPING = {1: "A", 2: "B"}
FS = 64
LAG_MS = 250
DECISION_WINDOW_SEC = 10
FIXED_LAMBDA = 1000.0
BP_LOWCUT = 1.0
BP_HIGHCUT = 8.0

CHANNEL_SUBSETS = {
    "2-Channel (Cz, Pz)": [47, 30],
    "4-Channel (Cz, Pz, C3, C4)": [47, 30, 12, 49],
    "8-Channel (Cz, C3, C4, CPz, CP3, CP4, Pz, Fz)": [47, 12, 49, 31, 17, 53, 30, 37]
}

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2, axis=0):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data, axis=axis)
    return y

def evaluate_windows(pred, ea, eb, window_samples):
    num_correct = 0.0
    num_total = 0
    start = 0
    while start + window_samples <= len(pred):
        end = start + window_samples
        p = pred[start:end]
        e_a = ea[start:end]
        e_b = eb[start:end]
        
        ca = np.corrcoef(p, e_a)[0, 1]
        cb = np.corrcoef(p, e_b)[0, 1]
        
        if ca > cb:
            num_correct += 1.0
        elif ca == cb:
            num_correct += 0.5
            
        num_total += 1
        start += window_samples
        
    return num_correct, num_total

def evaluate_subset(name, channels):
    print(f"\n===============================================================")
    print(f" EVALUATING: {name}")
    print(f"===============================================================")
    
    paths = subject_files()
    subject_examples = {}
    for p in paths:
        subject_examples[str(p)] = load_subject_examples(p)
        
    num_lags = int(FS * (LAG_MS / 1000.0)) + 1
    window_samples = DECISION_WINDOW_SEC * FS
    
    total_normal_corr = 0
    total_wins = 0
    
    for held_out_path in paths:
        held_out_key = str(held_out_path)
        exs = subject_examples[held_out_key]
        
        preprocessed_trials = []
        for ex in exs:
            eeg = ex.eeg[:, channels].T
            eeg = butter_bandpass_filter(eeg, BP_LOWCUT, BP_HIGHCUT, FS, axis=1)
            wav_a = butter_bandpass_filter(ex.wav_a.reshape(-1, 1), BP_LOWCUT, BP_HIGHCUT, FS, axis=0).ravel()
            wav_b = butter_bandpass_filter(ex.wav_b.reshape(-1, 1), BP_LOWCUT, BP_HIGHCUT, FS, axis=0).ravel()
            
            shifted = np.vstack([eeg.T[num_lags-1:, :], np.zeros((num_lags-1, len(channels)))])
            x_feat = np.zeros((shifted.shape[0], len(channels) * num_lags))
            for lag in range(num_lags):
                shift_amt = lag
                if shift_amt == 0:
                    x_feat[:, lag*len(channels):(lag+1)*len(channels)] = shifted
                else:
                    x_feat[:-shift_amt, lag*len(channels):(lag+1)*len(channels)] = shifted[shift_amt:, :]
                    
            mlen = min(x_feat.shape[0], len(wav_a), len(wav_b))
            x_feat = x_feat[:mlen]
            wav_a = wav_a[:mlen]
            wav_b = wav_b[:mlen]
            
            preprocessed_trials.append({
                'x': x_feat,
                'ea': wav_a,
                'eb': wav_b
            })
            
        subj_normal_corr = 0
        subj_total_wins = 0
        
        for test_idx in range(len(preprocessed_trials)):
            x_train_list = []
            y_train_list = []
            
            for i, p_trial in enumerate(preprocessed_trials):
                if i == test_idx:
                    continue
                x_train_list.append(p_trial['x'])
                y_train_list.append(p_trial['ea'])
                
            X_train = np.vstack(x_train_list)
            Y_train = np.concatenate(y_train_list)
            
            mu_x = X_train.mean(axis=0, keepdims=True)
            std_x = X_train.std(axis=0, keepdims=True) + 1e-12
            X_train = (X_train - mu_x) / std_x
            
            mu_y = Y_train.mean()
            std_y = Y_train.std() + 1e-12
            Y_train = (Y_train - mu_y) / std_y
            
            cov_xx = X_train.T @ X_train
            cov_xy = X_train.T @ Y_train
            reg_matrix = FIXED_LAMBDA * np.eye(X_train.shape[1])
            final_weights = np.linalg.solve(cov_xx + reg_matrix, cov_xy)
            
            p_test = preprocessed_trials[test_idx]
            x_test_feat = p_test['x']
            ea_test = p_test['ea']
            eb_test = p_test['eb']
            
            x_test_feat = (x_test_feat - mu_x) / std_x
            pred = x_test_feat @ final_weights
            
            nc_norm, ntot = evaluate_windows(pred, ea_test, eb_test, window_samples)
            subj_normal_corr += nc_norm
            subj_total_wins += ntot
            
        acc_norm = subj_normal_corr / subj_total_wins
        print(f"[{held_out_path.stem}] Acc: {acc_norm*100:.2f}%")
        
        total_normal_corr += subj_normal_corr
        total_wins += subj_total_wins
        
    final_acc = total_normal_corr / total_wins
    print(f"\n=> FINAL {name} ACCURACY: {final_acc*100:.2f}%\n")
    return final_acc

def main():
    for name, channels in CHANNEL_SUBSETS.items():
        evaluate_subset(name, channels)

if __name__ == "__main__":
    main()
