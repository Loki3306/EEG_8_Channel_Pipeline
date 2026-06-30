import numpy as np
import torch
from pathlib import Path

def safe_corr_np(x: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> float:
    """Compute Pearson correlation robustly."""
    x_mean = x.mean()
    y_mean = y.mean()
    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
    return float(num / (den + eps))

def evaluate_trial_majority_vote(predicted: np.ndarray, wav_a: np.ndarray, wav_b: np.ndarray, window_seconds: int = 10, hop_seconds: float = 1.0, fs: int = 64):
    """
    Evaluates Trial Accuracy and Window Accuracy based on 10s windows.
    Returns: (trial_correct_bool, total_windows, correct_windows)
    """
    win_samples = int(window_seconds * fs)
    hop_samples = int(hop_seconds * fs)
    
    if win_samples >= predicted.shape[0]:
        c_a = safe_corr_np(predicted, wav_a)
        c_b = safe_corr_np(predicted, wav_b)
        return c_a > c_b, 1, 1 if c_a > c_b else 0
        
    correct_windows = 0
    total_windows = 0
    
    for start in range(0, predicted.shape[0] - win_samples + 1, hop_samples):
        stop = start + win_samples
        c_a = safe_corr_np(predicted[start:stop], wav_a[start:stop])
        c_b = safe_corr_np(predicted[start:stop], wav_b[start:stop])
        if c_a > c_b:
            correct_windows += 1
        total_windows += 1
        
    if total_windows == 0:
        return False, 0, 0
        
    trial_correct = (correct_windows > total_windows / 2.0)
    return trial_correct, total_windows, correct_windows

def normalize_eeg(eeg: torch.Tensor) -> torch.Tensor:
    """Standard Z-score normalization for EEG per channel."""
    return (eeg - eeg.mean(dim=-1, keepdim=True)) / (eeg.std(dim=-1, keepdim=True) + 1e-8)

def normalize_audio(audio: torch.Tensor) -> torch.Tensor:
    """Standard Z-score normalization for audio envelope."""
    return (audio - audio.mean(dim=-1, keepdim=True)) / (audio.std(dim=-1, keepdim=True) + 1e-8)
