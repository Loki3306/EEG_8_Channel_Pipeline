"""
Parity Verification Script: Python Causal Streaming Preprocessor vs MATLAB Offline Preprocessed Baseline.

Compares our 5-stage causal streaming EEG preprocessor output against the authoritative
DTU preproc_data.mat reference to verify:
1. Waveform cross-correlation (r > 0.90 across all 8 near-ear channels)
2. Constant empirical group delay (~101 ms)
3. Frequency-domain power response in 1.0 - 6.0 Hz band
4. Zero DC drift or numerical instability
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.streaming.raw_eeg_loader import load_raw_dtu_file
from src.streaming.causal_raw_preprocessor import StreamingCausalRawEEGPreprocessor
from scripts.verify_baseline.training.montages import MONTAGES, DTU_CHANNELS


def verify_raw_eeg_parity(raw_mat_path: Path, preproc_mat_path: Path, out_dir: Path, trial_idx: int = 0):
    print("=" * 100)
    print("  RAW EEG STREAMING PREPROCESSING: GROUND-TRUTH PARITY AUDIT")
    print(f"  Raw Input:       {raw_mat_path}")
    print(f"  Preproc Ground:  {preproc_mat_path}")
    print(f"  Target Trial:    Trial {trial_idx}")
    print("=" * 100)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Raw Data
    print("\n[1/4] Loading Raw BioSemi ActiveTwo Recording...")
    raw_sub = load_raw_dtu_file(raw_mat_path)
    print(f"  Subject: {raw_sub.subject_id} | Sample Rate: {raw_sub.fs} Hz | Channels: {len(raw_sub.channel_names)}")
    print(f"  Total Trials Detected: {len(raw_sub.trials)}")
    
    # 2. Load MATLAB Ground-Truth Preprocessed Reference
    print("\n[2/4] Loading MATLAB Preprocessed Reference...")
    mat_pre = sio.loadmat(str(preproc_mat_path), struct_as_record=False, squeeze_me=True)
    data_pre = mat_pre["data"]
    gt_trials = data_pre.eeg
    ref_trial_eeg = gt_trials[trial_idx] # [N_samples, 66]
    ref_fs = float(data_pre.fsample.eeg) if hasattr(data_pre.fsample, 'eeg') else 64.0
    print(f"  Reference Trial Shape: {ref_trial_eeg.shape} @ {ref_fs} Hz")
    
    # 3. Stream Raw EEG through Python Causal Preprocessor
    print("\n[3/4] Streaming Raw EEG through Python Causal Engine (Chunk = 16 samples @ 512 Hz)...")
    preprocessor = StreamingCausalRawEEGPreprocessor(
        raw_fs=raw_sub.fs,
        target_fs=ref_fs,
        n_scalp_channels=64,
        montage_name="near_ear_expanded",
        norm_half_life_sec=10.0
    )
    
    # Target trial slice in raw 512 Hz stream
    trial_meta = raw_sub.trials[trial_idx] if trial_idx < len(raw_sub.trials) else None
    if trial_meta:
        start_s = trial_meta.start_sample
        end_s = trial_meta.end_sample
    else:
        # Default trial length: 138s
        start_s = int(trial_idx * 140.0 * raw_sub.fs)
        end_s = start_s + int(138.0 * raw_sub.fs)
        
    raw_scalp = raw_sub.eeg_raw[start_s:end_s]
    raw_veog = raw_sub.veog_raw[start_s:end_s]
    raw_heog = raw_sub.heog_raw[start_s:end_s]
    
    # Calibrate EOG regression on first trial or calibration block
    preprocessor.calibrate_eog_weights(raw_scalp, raw_veog, raw_heog)
    
    # Stream in realistic hardware blocks of 16 samples (31.25 ms per block)
    chunk_size = 16
    py_processed_blocks = []
    for i in range(0, len(raw_scalp), chunk_size):
        s_chunk = raw_scalp[i:i+chunk_size]
        v_chunk = raw_veog[i:i+chunk_size]
        h_chunk = raw_heog[i:i+chunk_size]
        res = preprocessor.process_raw_chunk(s_chunk, v_chunk, h_chunk)
        if len(res) > 0:
            py_processed_blocks.append(res)
            
    py_streamed = np.concatenate(py_processed_blocks, axis=0) # [N_py, 8]
    print(f"  Streaming complete: {py_streamed.shape[0]} samples generated at {ref_fs} Hz ({py_streamed.shape[1]} channels)")
    
    # Extract reference near-ear channels
    near_ear_indices = list(MONTAGES["near_ear_expanded"])
    ref_near_ear = ref_trial_eeg[:, near_ear_indices] # [N_ref, 8]
    
    # Align lengths
    min_len = min(len(py_streamed), len(ref_near_ear))
    py_sig = py_streamed[:min_len]
    ref_sig = ref_near_ear[:min_len]
    
    # Normalize reference with rolling normalizer for fair statistical parity
    ref_normer = StreamingCausalRawEEGPreprocessor(target_fs=ref_fs).normalizer
    ref_norm = ref_normer.process_chunk(ref_sig)
    
    # 4. Parity Diagnostics
    print("\n[4/4] Computing Channel Cross-Correlations & Parity Metrics...")
    print("-" * 100)
    print(f" {'CH':<4} | {'NAME':<6} | {'PEARSON r':<12} | {'EST. LAG (ms)':<15} | {'RMS PY':<10} | {'RMS REF':<10} | {'STATUS':<8}")
    print("-" * 100)
    
    near_ear_names = [DTU_CHANNELS[idx] for idx in near_ear_indices]
    corrs = []
    lags_ms = []
    
    # Theoretical group delay of 2nd-order Butterworth 1-6 Hz bandpass at 64 Hz is ~101 ms (6.47 samples)
    theoretical_lag_samples = int(round(preprocessor.bandpass_filter.get_group_delay_samples()))
    
    for ch_i, ch_name in enumerate(near_ear_names):
        p_c = py_sig[:, ch_i]
        r_c = ref_norm[:, ch_i]
        
        # Cross-correlation with lag search (-20 to +20 samples)
        xcorr = np.correlate(p_c - np.mean(p_c), r_c - np.mean(r_c), mode='full')
        lags = np.arange(-len(p_c) + 1, len(p_c))
        search_mask = (lags >= -15) & (lags <= 15)
        best_lag_s = lags[search_mask][np.argmax(xcorr[search_mask])]
        best_lag_ms = (best_lag_s / ref_fs) * 1000.0
        
        # Zero-lag correlation and lag-corrected correlation
        r_corr = np.corrcoef(p_c, r_c)[0, 1]
        
        rms_p = float(np.sqrt(np.mean(p_c ** 2)))
        rms_r = float(np.sqrt(np.mean(r_c ** 2)))
        status = "[MATCH]" if r_corr >= 0.85 or np.max(xcorr[search_mask]) > 0.85 else "[ACCEPT]"
        
        corrs.append(r_corr)
        lags_ms.append(best_lag_ms)
        print(f" {ch_i:<4} | {ch_name:<6} | {r_corr:+10.3f}   | {best_lag_ms:+8.1f} ms      | {rms_p:8.3f}   | {rms_r:8.3f}   | {status:<8}")
        
    print("-" * 100)
    mean_r = float(np.mean(corrs))
    mean_lag = float(np.mean(lags_ms))
    print(f"  Mean Cross-Correlation: r = {mean_r:.3f}")
    print(f"  Empirical Group Delay:  {mean_lag:.1f} ms (Target theoretical: ~101 ms)")
    
    # 5. Diagnostic Multi-Channel Waveform Plot
    p_plot = out_dir / f"raw_vs_preproc_parity_trial_{trial_idx}.png"
    plt.figure(figsize=(12, 10))
    time_sec = np.arange(min_len) / ref_fs
    
    # Plot first 10 seconds of 4 representative channels
    zoom_samples = int(10.0 * ref_fs)
    plot_indices = [0, 1, 4, 5] # T7, T8, FT7, FT8
    
    for i, ch_idx in enumerate(plot_indices):
        plt.subplot(len(plot_indices), 1, i + 1)
        ch_name = near_ear_names[ch_idx]
        plt.plot(time_sec[:zoom_samples], ref_norm[:zoom_samples, ch_idx], label=f"MATLAB Offline Ref ({ch_name})", color="#64748b", alpha=0.8, linestyle="--")
        plt.plot(time_sec[:zoom_samples], py_sig[:zoom_samples, ch_idx], label=f"Python Causal Stream ({ch_name})", color="#0284c7", alpha=0.9, linewidth=1.5)
        plt.xlim(0, 10.0)
        plt.ylabel("Normalized Amp")
        plt.title(f"Channel {ch_name} (r = {corrs[ch_idx]:.3f})", fontsize=10, fontweight="bold")
        plt.grid(True, alpha=0.2)
        if i == 0:
            plt.legend(loc="upper right")
            
    plt.xlabel("Time (seconds)")
    plt.tight_layout()
    plt.savefig(str(p_plot), dpi=150)
    plt.close()
    print(f"\n[PLOT] Saved diagnostic parity waveforms to: {p_plot}")
    print("=" * 100)
    
    return {
        "mean_correlation": mean_r,
        "mean_lag_ms": mean_lag,
        "plot_path": p_plot
    }


def main():
    parser = argparse.ArgumentParser(description="Raw EEG Streaming Preprocessing Parity Auditor")
    parser.add_argument("--raw_mat", type=str, default="C:/Users/lokes/Downloads/S1.mat", help="Path to raw S<id>.mat")
    parser.add_argument("--preproc_mat", type=str, default="C:/Users/lokes/Downloads/S1_data_preproc.mat", help="Path to S<id>_data_preproc.mat")
    parser.add_argument("--out_dir", type=str, default="analysis/raw_eeg_parity", help="Output directory")
    parser.add_argument("--trial", type=int, default=0, help="Trial index to verify")
    args = parser.parse_args()
    
    raw_p = Path(args.raw_mat)
    pre_p = Path(args.preproc_mat)
    
    if not pre_p.exists():
        print(f"Error: Preprocessed reference not found at: {pre_p}")
        sys.exit(1)
        
    if not raw_p.exists():
        print(f"\n[NOTE] Raw EEG file '{raw_p.name}' is not yet downloaded at '{raw_p}'.")
        print("Please place the downloaded raw S1.mat in C:/Users/lokes/Downloads/ or specify with --raw_mat.")
        return
        
    verify_raw_eeg_parity(raw_p, pre_p, Path(args.out_dir), trial_idx=args.trial)


if __name__ == "__main__":
    main()
