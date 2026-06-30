import numpy as np
import torch
from pathlib import Path
from .utils import normalize_eeg, normalize_audio, evaluate_trial_majority_vote, safe_corr_np

def get_base_metrics(model, test_trials, device):
    """Run unmodified baseline to get metrics."""
    if not test_trials:
        return {}
        
    model.eval()
    t_corr, total_w, w_corr = 0, 0, 0
    margins, p_att, p_unatt = [], [], []
    
    with torch.no_grad():
        for t in test_trials:
            eeg = t["eeg"].unsqueeze(0).to(device)
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
            
    return {
        "Trial Accuracy": t_corr / len(test_trials),
        "Window Accuracy": w_corr / max(1, total_w),
        "Mean Margin": np.mean(margins),
        "Median Margin": np.median(margins),
        "Mean Pearson(att)": np.mean(p_att),
        "Mean Pearson(unatt)": np.mean(p_unatt)
    }

CHANNEL_NAMES = ['T7', 'C2', 'FT8', 'P7', 'CPz', 'Fp1', 'TP8', 'C3']

def run_leave_one_channel_out(model, test_trials, device, num_channels=8):
    """Shuffles the time axis of one channel at a time to measure importance (accuracy drop)."""
    base_metrics = get_base_metrics(model, test_trials, device)
    
    loco_results = {}
    importance_scores = []
    
    for ch in range(num_channels):
        model.eval()
        t_corr, total_w, w_corr = 0, 0, 0
        margins, p_att, p_unatt = [], [], []
        
        with torch.no_grad():
            for t in test_trials:
                eeg = t["eeg"].unsqueeze(0).to(device).clone()
                
                # Permutation Feature Importance: shuffle the time dimension
                idx = torch.randperm(eeg.shape[-1], device=device)
                eeg[:, ch, :] = eeg[:, ch, idx]
                
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
        
        # Importance is the drop in Trial Accuracy and Margin
        acc_drop = base_metrics["Trial Accuracy"] - t_acc
        margin_drop = base_metrics["Mean Margin"] - np.mean(margins)
        
        channel_name = CHANNEL_NAMES[ch] if ch < len(CHANNEL_NAMES) else f"Ch{ch}"
        
        loco_results[channel_name] = {
            "Trial Accuracy": t_acc,
            "Window Accuracy": w_acc,
            "Mean Margin": np.mean(margins),
            "Median Margin": np.median(margins),
            "Mean Pearson(att)": np.mean(p_att),
            "Mean Pearson(unatt)": np.mean(p_unatt),
            "Acc Drop": acc_drop,
            "Margin Drop": margin_drop
        }
        
        # A channel is "important" if dropping it causes a big margin drop
        importance_scores.append((ch, margin_drop))
        
    # Rank channels from most important (largest drop) to least important
    importance_scores.sort(key=lambda x: x[1], reverse=True)
    ranked_channels = [x[0] for x in importance_scores]
    
    return loco_results, ranked_channels

def run_progressive_ablation(model, test_trials, device, ranked_channels):
    """
    Keeps only Top N channels, ablates (shuffles) the rest.
    """
    num_channels = len(ranked_channels)
    n_configs = sorted(list(set([num_channels, max(1, num_channels - 2), max(1, num_channels - 4), max(1, num_channels - 6), 1])), reverse=True)
    ablation_results = {}
    
    for n in n_configs:
        keep_channels = ranked_channels[:n]
        
        model.eval()
        t_corr, total_w, w_corr = 0, 0, 0
        margins, p_att, p_unatt = [], [], []
        
        with torch.no_grad():
            for t in test_trials:
                eeg = t["eeg"].unsqueeze(0).to(device).clone()
                
                # Ablate (shuffle) all channels NOT in keep_channels
                for ch in range(eeg.shape[1]):
                    if ch not in keep_channels:
                        idx = torch.randperm(eeg.shape[-1], device=device)
                        eeg[:, ch, :] = eeg[:, ch, idx]
                        
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
        
        ablation_results[f"{n} Channels"] = {
            "Channels Kept": keep_channels,
            "Trial Accuracy": t_acc,
            "Window Accuracy": w_acc,
            "Mean Margin": np.mean(margins),
            "Median Margin": np.median(margins),
            "Mean Pearson(att)": np.mean(p_att),
            "Mean Pearson(unatt)": np.mean(p_unatt)
        }
        
    return ablation_results
