import numpy as np
import torch
from .utils import normalize_eeg, normalize_audio, evaluate_trial_majority_vote, safe_corr_np

def apply_fft_bandstop(eeg_tensor: torch.Tensor, lowcut: float, highcut: float, fs: int = 64) -> torch.Tensor:
    n_samples = eeg_tensor.shape[-1]
    fft_vals = torch.fft.rfft(eeg_tensor, dim=-1)
    freqs = torch.fft.rfftfreq(n_samples, d=1.0/fs).to(eeg_tensor.device)
    mask = torch.ones_like(freqs)
    mask[(freqs >= lowcut) & (freqs <= highcut)] = 0.0
    fft_vals_filtered = fft_vals * mask
    return torch.fft.irfft(fft_vals_filtered, n=n_samples, dim=-1)

def run_frequency_ablation(model, test_trials, device):
    """
    Evaluates model performance after ablating specific canonical EEG frequency bands.
    NOTE: This is an exploratory analysis. Inference-time spectral ablation via FFT masking 
    is not part of the standard validated evaluation protocol. Physiological conclusions 
    derived from these ablations should be considered exploratory rather than validated.
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
                eeg = apply_fft_bandstop(eeg, lowcut, highcut, fs=64)
                
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
