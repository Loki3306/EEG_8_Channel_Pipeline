import numpy as np
import torch
from .utils import normalize_eeg, normalize_audio

def safe_corr_pt(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """PyTorch version of Pearson correlation for autograd."""
    x_mean = x.mean(dim=-1, keepdim=True)
    y_mean = y.mean(dim=-1, keepdim=True)
    num = torch.sum((x - x_mean) * (y - y_mean), dim=-1)
    den = torch.sqrt(torch.sum((x - x_mean)**2, dim=-1) * torch.sum((y - y_mean)**2, dim=-1))
    return num / (den + eps)

def run_saliency_analysis(model, test_trials, device, window_seconds=10.0, fs=64):
    """
    Computes Input x Gradient saliency maps for the EEG input with respect to the
    positive classification margin (corr_att - corr_unatt).
    """
    win_samples = int(window_seconds * fs)
    
    # We will accumulate the absolute Input x Gradient saliency
    # Shape will be [Channels, Time] for the average 10s window
    total_saliency = torch.zeros(8, win_samples, device=device)
    num_windows = 0
    
    model.eval()
    
    # We need gradients for the input, so we must enable grad locally
    with torch.set_grad_enabled(True):
        for t in test_trials:
            eeg_full = t["eeg"].unsqueeze(0).to(device)
            wav_a_full = t["audio_a"].unsqueeze(0).to(device).mean(dim=1, keepdim=True)
            wav_b_full = t["audio_b"].unsqueeze(0).to(device).mean(dim=1, keepdim=True)
            
            eeg_full = normalize_eeg(eeg_full)
            wav_a_full = normalize_audio(wav_a_full)
            wav_b_full = normalize_audio(wav_b_full)
            
            min_len = min(eeg_full.shape[2], wav_a_full.shape[2], wav_b_full.shape[2])
            eeg_full = eeg_full[:,:,:min_len]
            wav_a_full = wav_a_full[:,:,:min_len]
            wav_b_full = wav_b_full[:,:,:min_len]
            
            for start in range(0, min_len - win_samples + 1, win_samples):
                eeg_chunk = eeg_full[:, :, start:start+win_samples].clone()
                eeg_chunk.requires_grad_(True)
                
                wa_chunk = wav_a_full[:, 0, start:start+win_samples]
                wb_chunk = wav_b_full[:, 0, start:start+win_samples]
                
                # Forward pass
                pred = model(eeg_chunk) # [Batch, Time]
                
                # Compute Margin
                ca = safe_corr_pt(pred, wa_chunk)
                cb = safe_corr_pt(pred, wb_chunk)
                margin = (ca - cb).mean() # Mean over batch (which is 1)
                
                # Backward pass to get gradients w.r.t eeg_chunk
                model.zero_grad()
                margin.backward()
                
                # Input x Gradient
                grad = eeg_chunk.grad.detach() # [1, Channels, Time]
                input_x_grad = (eeg_chunk.detach() * grad).abs().squeeze(0) # [Channels, Time]
                
                total_saliency += input_x_grad
                num_windows += 1

    # Average Saliency
    avg_saliency = total_saliency / max(1, num_windows)
    avg_saliency_np = avg_saliency.cpu().numpy()
    
    # Compute marginal importance
    channel_saliency = np.mean(avg_saliency_np, axis=1) # [Channels]
    temporal_saliency = np.mean(avg_saliency_np, axis=0) # [Time]
    
    return {
        "Saliency_Map": avg_saliency_np,
        "Channel_Saliency": channel_saliency,
        "Temporal_Saliency": temporal_saliency
    }
