import numpy as np
import torch
from .utils import normalize_eeg, normalize_audio, safe_corr_np

def run_temporal_occlusion(model, test_trials, device, window_seconds=10.0, step_seconds=0.250, fs=64):
    """
    Evaluates temporal occlusion within a 10s window.
    Breaks trials into 10s chunks, then sequentially masks (zeros) out 'step_seconds'
    slices within the 10s chunk to see which part of the 10s history is most important.
    """
    win_samples = int(window_seconds * fs)
    step_samples = int(step_seconds * fs)
    num_steps = win_samples // step_samples
    
    # Track the average margin drop for each temporal occlusion bin
    occlusion_margins = {i: [] for i in range(num_steps)}
    
    model.eval()
    with torch.no_grad():
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
            
            # Chunk the trial into non-overlapping 10s windows
            starts = list(range(0, min_len - win_samples + 1, win_samples))
            if not starts:
                continue
            if starts[-1] + win_samples < min_len:
                starts.append(min_len - win_samples) # Cover the tail
            
            for start in starts:
                eeg_chunk = eeg_full[:, :, start:start+win_samples]
                wa_chunk = wav_a_full[:, :, start:start+win_samples].squeeze(1).squeeze(0).cpu().numpy()
                wb_chunk = wav_b_full[:, :, start:start+win_samples].squeeze(1).squeeze(0).cpu().numpy()
                
                # Baseline prediction for this 10s chunk
                pred_base = model(eeg_chunk).squeeze(0).cpu().numpy()
                ca_base = safe_corr_np(pred_base, wa_chunk)
                cb_base = safe_corr_np(pred_base, wb_chunk)
                margin_base = ca_base - cb_base
                
                # Apply occlusion mask iteratively
                for step_idx in range(num_steps):
                    mask_start = step_idx * step_samples
                    mask_end = min(mask_start + step_samples, win_samples)
                    
                    eeg_masked = eeg_chunk.clone()
                    eeg_masked[:, :, mask_start:mask_end] = 0.0
                    
                    pred_masked = model(eeg_masked).squeeze(0).cpu().numpy()
                    ca_masked = safe_corr_np(pred_masked, wa_chunk)
                    cb_masked = safe_corr_np(pred_masked, wb_chunk)
                    margin_masked = ca_masked - cb_masked
                    
                    margin_drop = float(margin_base - margin_masked)
                    occlusion_margins[step_idx].append(margin_drop)
                    
    # Average the drops
    temporal_importance = []
    for step_idx in range(num_steps):
        time_sec = (step_idx * step_samples) / fs
        avg_drop = np.mean(occlusion_margins[step_idx])
        temporal_importance.append({
            "Time Start (s)": time_sec,
            "Time End (s)": time_sec + step_seconds,
            "Mean Margin Drop": avg_drop
        })
        
    return temporal_importance
