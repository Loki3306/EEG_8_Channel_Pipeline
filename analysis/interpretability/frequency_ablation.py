import numpy as np
import torch
from scipy.signal import butter, filtfilt
from .utils import normalize_eeg, normalize_audio, evaluate_trial_majority_vote, safe_corr_np

def apply_bandstop_filter(eeg_tensor: torch.Tensor, lowcut: float, highcut: float, fs: int = 64, order: int = 4) -> torch.Tensor:
    """
    Applies a Butterworth band-stop filter to the EEG tensor.
    eeg_tensor: [Batch, Channels, Time]
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Design band-stop filter
    b, a = butter(order, [low, high], btype='bandstop')
    
    # Convert tensor to numpy for scipy filtfilt
    eeg_np = eeg_tensor.cpu().numpy()
    
    # filtfilt applies filter forward and backward for zero phase shift
    filtered_np = filtfilt(b, a, eeg_np, axis=-1)
    
    # Ensure float32 and return as tensor
    return torch.from_numpy(filtered_np.astype(np.float32)).to(eeg_tensor.device)

def run_frequency_ablation(model, test_trials, device):
    """
    Evaluates model performance after ablating specific canonical EEG frequency bands.
    """
    bands = {
        "Delta (0.5-4Hz)": (0.5, 4.0),
        "Theta (4-8Hz)": (4.0, 8.0),
        "Alpha (8-13Hz)": (8.0, 13.0),
        "Beta (13-30Hz)": (13.0, 30.0)
    }
    
    freq_results = {}
    
    for band_name, (lowcut, highcut) in bands.items():
        model.eval()
        t_corr, total_w, w_corr = 0, 0, 0
        margins, p_att, p_unatt = [], [], []
        
        with torch.no_grad():
            for t in test_trials:
                eeg = t["eeg"].unsqueeze(0).to(device)
                
                # Apply bandstop filter to remove this frequency band
                eeg = apply_bandstop_filter(eeg, lowcut, highcut, fs=64)
                
                wav_a = t["audio_a"].unsqueeze(0).to(device).mean(dim=1, keepdim=True)
                wav_b = t["audio_b"].unsqueeze(0).to(device).mean(dim=1, keepdim=True)
                
                eeg = normalize_eeg(eeg)
                wav_a = normalize_audio(wav_a)
                wav_b = normalize_audio(wav_b)
                
                min_len = min(eeg.shape[2], wav_a.shape[2], wav_b.shape[2])
                eeg, wav_a, wav_b = eeg[:,:,:min_len], wav_a[:,:,:min_len], wav_b[:,:,:min_len]
                
                pred = model(eeg).squeeze(0).cpu().numpy()
                wa = wav_a.squeeze(1).squeeze(0).cpu().numpy()
                wb = wav_b.squeeze(1).squeeze(0).cpu().numpy()
                
                ca = safe_corr_np(pred, wa)
                cb = safe_corr_np(pred, wb)
                margin = ca - cb
                
                tok, nwin, cwin = evaluate_trial_majority_vote(pred, wa, wb)
                if tok: t_corr += 1
                total_w += nwin
                w_corr += cwin
                
                margins.append(float(margin))
                p_att.append(float(ca))
                p_unatt.append(float(cb))
                
        t_acc = t_corr / len(test_trials)
        w_acc = w_corr / max(1, total_w)
        
        freq_results[band_name] = {
            "Trial Accuracy": t_acc,
            "Window Accuracy": w_acc,
            "Mean Margin": np.mean(margins),
            "Median Margin": np.median(margins),
            "Mean Pearson(att)": np.mean(p_att),
            "Mean Pearson(unatt)": np.mean(p_unatt)
        }
        
    return freq_results
