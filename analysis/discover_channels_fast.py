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

FS = 64
LAG_MS = 250
FIXED_LAMBDA = 1000.0
BP_LOWCUT = 1.0
BP_HIGHCUT = 8.0

CHANNELS = list(range(64))
num_channels = 64

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2, axis=0):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data, axis=axis)
    return y

def main():
    print("===============================================================")
    print(" FAST DISCOVERY OF BEST EEG CHANNELS (<10 MIN)")
    print("===============================================================")
    print("Training a single global 64-channel Ridge model on all data...")
    
    paths = subject_files()
    subject_examples = {}
    for p in paths:
        subject_examples[str(p)] = load_subject_examples(p)
        
    num_lags = int(FS * (LAG_MS / 1000.0)) + 1
    
    all_x = []
    all_y = []
    
    for held_out_path in paths:
        held_out_key = str(held_out_path)
        exs = subject_examples[held_out_key]
        
        for ex in exs:
            eeg = ex.eeg[:, CHANNELS].T
            eeg = butter_bandpass_filter(eeg, BP_LOWCUT, BP_HIGHCUT, FS, axis=1)
            wav_a = butter_bandpass_filter(ex.wav_a.reshape(-1, 1), BP_LOWCUT, BP_HIGHCUT, FS, axis=0).ravel()
            
            shifted = np.vstack([eeg.T[num_lags-1:, :], np.zeros((num_lags-1, num_channels))])
            x_feat = np.zeros((shifted.shape[0], num_channels * num_lags))
            for lag in range(num_lags):
                shift_amt = lag
                if shift_amt == 0:
                    x_feat[:, lag*num_channels:(lag+1)*num_channels] = shifted
                else:
                    x_feat[:-shift_amt, lag*num_channels:(lag+1)*num_channels] = shifted[shift_amt:, :]
                    
            mlen = min(x_feat.shape[0], len(wav_a))
            x_feat = x_feat[:mlen]
            wav_a = wav_a[:mlen]
            
            all_x.append(x_feat)
            all_y.append(wav_a)
            
        print(f"Preprocessed {held_out_path.stem}...")

    print("Concatenating global dataset...")
    X_train = np.vstack(all_x)
    Y_train = np.concatenate(all_y)
    
    print("Normalizing...")
    mu_x = X_train.mean(axis=0, keepdims=True)
    std_x = X_train.std(axis=0, keepdims=True) + 1e-12
    X_train = (X_train - mu_x) / std_x
    
    mu_y = Y_train.mean()
    std_y = Y_train.std() + 1e-12
    Y_train = (Y_train - mu_y) / std_y
    
    print("Solving global Ridge Regression...")
    cov_xx = X_train.T @ X_train
    cov_xy = X_train.T @ Y_train
    reg_matrix = FIXED_LAMBDA * np.eye(X_train.shape[1])
    final_weights = np.linalg.solve(cov_xx + reg_matrix, cov_xy)
    
    # final_weights shape: (64 * 17,)
    # Extract importance by summing absolute weights across all lags for each channel
    channel_importance = np.zeros(num_channels)
    for c in range(num_channels):
        idx = [c + L * num_channels for L in range(num_lags)]
        channel_importance[c] = np.sum(np.abs(final_weights[idx]))
        
    # Sort channels by importance
    ranked_channels = np.argsort(channel_importance)[::-1]
    
    print("\n===============================================================")
    print(" CHANNEL RANKING (Highest to Lowest Contribution)")
    print("===============================================================")
    for rank, c in enumerate(ranked_channels):
        print(f"Rank {rank+1:02d} | Channel {c:02d} | Score: {channel_importance[c]:.4f}")
        
    print("\n===============================================================")
    print(" RECOMMENDED SUBSETS")
    print("===============================================================")
    print(f"Best 2-channel set: {list(ranked_channels[:2])}")
    print(f"Best 4-channel set: {list(ranked_channels[:4])}")
    print(f"Best 8-channel set: {list(ranked_channels[:8])}")

if __name__ == "__main__":
    main()
