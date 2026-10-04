import numpy as np
from typing import Dict, Any, Optional

def compute_sir_metrics(
    audio_att: np.ndarray,
    audio_unatt: np.ndarray,
    gain_att_traj: np.ndarray,
    gain_unatt_traj: np.ndarray
) -> Dict[str, float]:
    """
    Computes Signal-to-Interference Ratio (SIR) before and after AAD dynamic steering.
    
    audio_att: 1D array of attended speech samples
    audio_unatt: 1D array of unattended competing speech samples
    gain_att_traj: 1D array of applied linear gains on attended speech
    gain_unatt_traj: 1D array of applied linear gains on unattended speech
    
    Returns:
      sir_in_db: Baseline SIR in unsteered mixture
      sir_out_db: Output SIR after amplification and suppression
      delta_sir_db: Net acoustic separation improvement (dB)
    """
    min_len = min(len(audio_att), len(audio_unatt), len(gain_att_traj), len(gain_unatt_traj))
    att_in = audio_att[:min_len]
    unatt_in = audio_unatt[:min_len]
    g_att = gain_att_traj[:min_len]
    g_unatt = gain_unatt_traj[:min_len]
    
    e_att_in = float(np.sum(att_in ** 2)) + 1e-12
    e_unatt_in = float(np.sum(unatt_in ** 2)) + 1e-12
    sir_in_db = 10.0 * np.log10(e_att_in / e_unatt_in)
    
    att_out = att_in * g_att
    unatt_out = unatt_in * g_unatt
    
    e_att_out = float(np.sum(att_out ** 2)) + 1e-12
    e_unatt_out = float(np.sum(unatt_out ** 2)) + 1e-12
    sir_out_db = 10.0 * np.log10(e_att_out / e_unatt_out)
    
    delta_sir_db = sir_out_db - sir_in_db
    
    return {
        "sir_in_db": sir_in_db,
        "sir_out_db": sir_out_db,
        "delta_sir_db": delta_sir_db
    }

def compute_headroom_metrics(stereo_audio: np.ndarray) -> Dict[str, float]:
    """
    Computes peak amplitude, RMS energy, and dynamic headroom in dBFS.
    """
    peak = float(np.max(np.abs(stereo_audio)))
    rms = float(np.sqrt(np.mean(stereo_audio ** 2)))
    
    peak_dbfs = 20.0 * np.log10(max(peak, 1e-6))
    rms_dbfs = 20.0 * np.log10(max(rms, 1e-6))
    headroom_db = -peak_dbfs # Headroom to 0 dBFS
    clipping_rate = float(np.mean(np.abs(stereo_audio) >= 0.999) * 100.0)
    
    return {
        "peak_dbfs": peak_dbfs,
        "rms_dbfs": rms_dbfs,
        "headroom_db": headroom_db,
        "clipping_rate_pct": clipping_rate
    }

def compute_stoi_intelligibility(
    clean_ref: np.ndarray,
    processed_audio: np.ndarray,
    fs: int = 16000
) -> float:
    """
    Computes Short-Time Objective Intelligibility (STOI, Taal et al. 2011).
    Attempts pystoi first; falls back to an embedded 1/3-octave envelope correlation.
    
    Returns:
      score: float between 0.0 (unintelligible) and 1.0 (perfect intelligibility)
    """
    # Mono conversion if stereo
    if clean_ref.ndim > 1:
        clean_ref = np.mean(clean_ref, axis=0)
    if processed_audio.ndim > 1:
        processed_audio = np.mean(processed_audio, axis=0)
        
    min_len = min(len(clean_ref), len(processed_audio))
    clean = clean_ref[:min_len].astype(np.float64)
    proc = processed_audio[:min_len].astype(np.float64)
    
    try:
        import pystoi
        return float(pystoi.stoi(clean, proc, fs, extended=False))
    except ImportError:
        pass
        
    # Standalone fallback STOI implementation
    # Resample / frame into 25 ms windows with 50% overlap
    win_len = int(0.025 * fs) # 25 ms
    hop_len = int(0.0125 * fs) # 12.5 ms
    
    n_frames = (min_len - win_len) // hop_len
    if n_frames <= 0:
        return 0.5
        
    window = np.hanning(win_len)
    clean_frames = np.array([clean[i*hop_len : i*hop_len + win_len] * window for i in range(n_frames)])
    proc_frames = np.array([proc[i*hop_len : i*hop_len + win_len] * window for i in range(n_frames)])
    
    # Octave band energy estimation
    clean_spec = np.abs(np.fft.rfft(clean_frames, axis=-1))
    proc_spec = np.abs(np.fft.rfft(proc_frames, axis=-1))
    
    # 15 1/3-octave bands from 150 Hz to 4300 Hz
    n_bins = clean_spec.shape[-1]
    freqs = np.fft.rfftfreq(win_len, 1.0 / fs)
    
    # Compute Pearson correlation across frames per frequency band
    corrs = []
    # Band division (approx 15 sub-bands)
    band_edges = np.geomspace(150, min(fs/2 - 100, 4500), num=16)
    for b in range(15):
        idx = np.where((freqs >= band_edges[b]) & (freqs < band_edges[b+1]))[0]
        if len(idx) == 0:
            continue
        c_band = np.sum(clean_spec[:, idx], axis=-1)
        p_band = np.sum(proc_spec[:, idx], axis=-1)
        
        c_std = np.std(c_band)
        p_std = np.std(p_band)
        if c_std > 1e-6 and p_std > 1e-6:
            r = float(np.corrcoef(c_band, p_band)[0, 1])
            if not np.isnan(r):
                corrs.append(np.clip(r, 0.0, 1.0))
                
    if not corrs:
        return 0.5
    return float(np.mean(corrs))

def evaluate_audio_steering_trial(
    audio_a: np.ndarray,
    audio_b: np.ndarray,
    render_dict: Dict[str, Any],
    ground_truth: str = "A",
    decisions: Optional[list] = None,
    step_sec: float = 0.5,
) -> Dict[str, Any]:
    """
    Evaluates acoustic separation, speech intelligibility, and controller metrics for a full trial.
    """
    steered = render_dict["steered_binaural"]
    mixture = render_dict["raw_mixture"]
    clean_ref = render_dict["clean_attended_reference"]
    g_a = render_dict["gain_trajectory_a"]
    g_b = render_dict["gain_trajectory_b"]
    fs = render_dict["fs"]
    
    if ground_truth == "A":
        att_audio = audio_a
        unatt_audio = audio_b
        g_att = g_a
        g_unatt = g_b
    else:
        att_audio = audio_b
        unatt_audio = audio_a
        g_att = g_b
        g_unatt = g_a
        
    sir_metrics = compute_sir_metrics(att_audio, unatt_audio, g_att, g_unatt)
    headroom_metrics = compute_headroom_metrics(steered)
    
    # STOI intelligibility of steered output vs unsteered raw mixture
    stoi_steered = compute_stoi_intelligibility(clean_ref, steered, fs)
    stoi_mixture = compute_stoi_intelligibility(clean_ref, mixture, fs)
    delta_stoi = stoi_steered - stoi_mixture
    
    # Gain statistics
    mean_gain_att_db = float(np.mean(20.0 * np.log10(np.maximum(g_att, 1e-6))))
    mean_gain_unatt_db = float(np.mean(20.0 * np.log10(np.maximum(g_unatt, 1e-6))))
    mean_contrast_db = mean_gain_att_db - mean_gain_unatt_db
    
    metrics = {
        "sir_in_db": sir_metrics["sir_in_db"],
        "sir_out_db": sir_metrics["sir_out_db"],
        "delta_sir_db": sir_metrics["delta_sir_db"],
        "stoi_steered": stoi_steered,
        "stoi_mixture": stoi_mixture,
        "delta_stoi": delta_stoi,
        "mean_gain_att_db": mean_gain_att_db,
        "mean_gain_unatt_db": mean_gain_unatt_db,
        "mean_contrast_db": mean_contrast_db,
        "peak_dbfs": headroom_metrics["peak_dbfs"],
        "rms_dbfs": headroom_metrics["rms_dbfs"],
        "headroom_db": headroom_metrics["headroom_db"],
        "clipping_rate_pct": headroom_metrics["clipping_rate_pct"],
    }
    
    # Model decision metrics if provided
    if decisions is not None:
        dec_arr = np.array(decisions)
        gt_arr = np.array([ground_truth] * len(dec_arr))
        
        correct_mask = (dec_arr == gt_arr)
        hold_mask = (dec_arr == "HOLD")
        
        acc = float(np.mean(correct_mask) * 100.0)
        hold_rate = float(np.mean(hold_mask) * 100.0)
        boost_cov = float(np.mean(g_att >= 1.0) * 100.0)
        
        # False switches
        flips = 0
        last_s = None
        for s in dec_arr:
            if s != "HOLD":
                if last_s is not None and s != last_s:
                    flips += 1
                last_s = s
        duration_min = (len(dec_arr) * step_sec) / 60.0
        fsw_per_min = float(flips / max(duration_min, 1e-4))
        
        metrics.update({
            "decision_accuracy_pct": acc,
            "hold_rate_pct": hold_rate,
            "boost_coverage_pct": boost_cov,
            "false_switches_per_min": fsw_per_min
        })
        
    return metrics
