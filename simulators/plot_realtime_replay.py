import argparse
import sys
import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

VERIFY_ROOT = REPO_ROOT / "scripts" / "verify_baseline"
if str(VERIFY_ROOT) not in sys.path:
    sys.path.insert(0, str(VERIFY_ROOT))

from src.streaming.pipeline import StreamingAADPipeline
from scripts.verify_baseline.models.catcn import CATCNDirectDecoder
from scripts.verify_baseline.training.montages import MONTAGES
from scripts.verify_baseline.training.train_matchnet_wavlm import get_mapping_data, prepare_dataset, FS
from scripts.verify_baseline.baselines.ridge_aad import load_subject_examples, subject_files

def generate_telemetry_plot(
    subject: str = "S1_data_preproc",
    montage: str = "near_ear_expanded",
    trial_idx: int = 0,
    checkpoint: str = "",
    output_png: str = "streaming_bci_telemetry.png",
    window_sec: float = 5.0,
    step_sec: float = 0.5
):
    # 1. Setup Model
    montage_channels = MONTAGES[montage]
    n_ch = len(montage_channels)
    model = CATCNDirectDecoder(eeg_channels=n_ch, audio_channels=1, hidden_dim=64, max_lag_samples=8)
    
    ckpt_path = checkpoint
    if not ckpt_path and Path("/kaggle/working/catcn_deployment_weights.pt").exists():
        ckpt_path = "/kaggle/working/catcn_deployment_weights.pt"
        
    if ckpt_path and Path(ckpt_path).exists():
        print(f"[MODEL] Loading trained checkpoint from: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state.get("model_state_dict", state.get("state_dict", state)))
        print("[MODEL] Checkpoint loaded successfully!")
    else:
        print("[MODEL WARNING] Running with initialized weights.")
        
    model.eval()

    # 2. Load Genuine DTU Data
    files = subject_files()
    target_files = [f for f in files if subject in f.name]
    if not target_files:
        raise FileNotFoundError(f"Could not find DTU subject file for {subject} in DATA_DIR.")
        
    print(f"[DATA] Loading genuine DTU recording: {target_files[0].name}...")
    mapping, envelopes = get_mapping_data("gammatone")
    test_exs = list(load_subject_examples(target_files[0]))
    
    _, YA_all, YB_all = prepare_dataset(test_exs, montage_channels, 1.0, 6.0, subject, mapping, envelopes)
    
    t_idx = min(trial_idx, len(test_exs) - 1, len(YA_all) - 1)
    raw_eeg = test_exs[t_idx].eeg[:, montage_channels].astype(np.float32)
    ya = YA_all[t_idx].mean(axis=0).squeeze() if YA_all[t_idx].ndim > 1 else YA_all[t_idx].squeeze()
    yb = YB_all[t_idx].mean(axis=0).squeeze() if YB_all[t_idx].ndim > 1 else YB_all[t_idx].squeeze()
    
    min_len = min(len(raw_eeg), len(ya), len(yb))
    raw_eeg = raw_eeg[:min_len]
    ya = ya[:min_len]
    yb = yb[:min_len]
    total_sec = min_len / FS
    print(f"[DATA] Replaying Trial {t_idx} ({total_sec:.1f} s)...")

    # 3. Initialize Streaming Pipeline
    pipeline = StreamingAADPipeline(
        model=model,
        n_eeg_channels=n_ch,
        fs=FS,
        raw_audio_input=False,
        window_sec=window_sec,
        step_sec=step_sec,
        engine_mode="torchscript",
        decision_alpha=0.7,
        decision_threshold=0.25,
        n_confirm=2,
        boost_db=6.0
    )

    # 4. Stream and Collect Telemetry
    timestamps = []
    logits_a = []
    logits_b = []
    deltas = []
    smoothed = []
    gains_a = []
    gains_b = []
    states = []
    switches = []
    latencies = []

    chunk_samples = 16
    idx = 0
    while idx < min_len:
        end_idx = min(idx + chunk_samples, min_len)
        chunk_e = raw_eeg[idx:end_idx]
        chunk_a = ya[idx:end_idx]
        chunk_b = yb[idx:end_idx]
        
        telemetry = pipeline.feed_sample_block(chunk_e, chunk_a, chunk_b)
        if telemetry is not None:
            timestamps.append(telemetry["timestamp_sec"])
            logits_a.append(telemetry["logit_a"])
            logits_b.append(telemetry["logit_b"])
            deltas.append(telemetry["raw_delta"])
            smoothed.append(telemetry["smoothed_score"])
            
            ga_db = 20.0 * np.log10(max(1e-3, telemetry["gain_a"]))
            gb_db = 20.0 * np.log10(max(1e-3, telemetry["gain_b"]))
            gains_a.append(ga_db)
            gains_b.append(gb_db)
            
            states.append(telemetry["attended_stream"])
            switches.append(telemetry["switched"])
            latencies.append(telemetry["compute_ms"])
            
        idx = end_idx

    timestamps = np.array(timestamps)
    logits_a = np.array(logits_a)
    logits_b = np.array(logits_b)
    deltas = np.array(deltas)
    smoothed = np.array(smoothed)
    gains_a = np.array(gains_a)
    gains_b = np.array(gains_b)
    latencies = np.array(latencies)

    # 5. Create 4-Panel Publication-Quality Plot
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True, gridspec_kw={"height_ratios": [1.2, 1.4, 1.2, 0.8]})
    
    # Palette
    color_a = "#1f77b4" # Blue for Stream A
    color_b = "#ff7f0e" # Orange for Stream B
    color_neut = "#7f7f7f"
    
    # Background state shading on all panels
    for i in range(len(timestamps) - 1):
        t_start = timestamps[i] - step_sec / 2.0
        t_end = timestamps[i+1] - step_sec / 2.0
        st = states[i]
        bg_col = "#e6f2ff" if st == "A" else ("#fff2e6" if st == "B" else "#f0f0f0")
        for ax in axes:
            ax.axvspan(t_start, t_end, color=bg_col, alpha=0.35, zorder=0)

    # PANEL 1: Neural Correlation Logits
    ax1 = axes[0]
    ax1.plot(timestamps, logits_a, label="Neural Logit A (Speaker A)", color=color_a, linewidth=2.0)
    ax1.plot(timestamps, logits_b, label="Neural Logit B (Speaker B)", color=color_b, linewidth=2.0)
    ax1.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.set_ylabel("Correlation Logit", fontsize=11, fontweight="bold")
    ax1.set_title(f"Real-Time Streaming BCI Telemetry | Patient: {subject} | Trial: {t_idx} (8-Ch Peri-Auricular EEG)", fontsize=13, fontweight="bold", pad=12)
    ax1.legend(loc="upper right", frameon=True, framealpha=0.9)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # PANEL 2: Decision Margin & EMA Smoothed Trajectory
    ax2 = axes[1]
    ax2.plot(timestamps, deltas, label="Raw Margin Δ_t (Logit A - Logit B)", color="#a6cee3", linestyle="--", alpha=0.7, linewidth=1.2)
    ax2.plot(timestamps, smoothed, label="EMA Smoothed Trajectory S_t (α=0.7)", color="#08519c", linewidth=2.5)
    ax2.axhline(+0.25, color=color_a, linestyle=":", linewidth=1.5, label="Lock-on Stream A Threshold (+0.25)")
    ax2.axhline(-0.25, color=color_b, linestyle=":", linewidth=1.5, label="Lock-on Stream B Threshold (-0.25)")
    ax2.axhline(0.0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
    
    # Mark speaker switches
    switch_indices = np.where(switches)[0]
    for s_idx in switch_indices:
        t_sw = timestamps[s_idx]
        val_sw = smoothed[s_idx]
        ax2.scatter(t_sw, val_sw, color="red", s=90, zorder=5, marker="*")
        ax2.annotate("SWITCH", xy=(t_sw, val_sw), xytext=(t_sw + 0.3, val_sw + (0.25 if val_sw > 0 else -0.35)),
                     arrowprops=dict(facecolor="red", arrowstyle="->", lw=1.2), fontsize=9, fontweight="bold", color="red")
        
    ax2.set_ylabel("Decision Score (S_t)", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper left", frameon=True, framealpha=0.9, ncol=2)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # PANEL 3: Active Hearing Aid Attenuation Gains (dB)
    ax3 = axes[2]
    ax3.plot(timestamps, gains_a, label="Hearing Aid Gain A (dB)", color=color_a, linewidth=2.2)
    ax3.plot(timestamps, gains_b, label="Hearing Aid Gain B (dB)", color=color_b, linewidth=2.2)
    ax3.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax3.axhline(-6.0, color="red", linestyle=":", linewidth=1.0, alpha=0.7, label="Max Suppression (-6.0 dB)")
    ax3.set_ylabel("Gain (dB)", fontsize=11, fontweight="bold")
    ax3.set_ylim(-6.8, 0.5)
    ax3.legend(loc="lower right", frameon=True, framealpha=0.9)
    ax3.grid(True, linestyle=":", alpha=0.6)

    # PANEL 4: Edge Compute Latency
    ax4 = axes[3]
    ax4.plot(timestamps, latencies, color="#2ca02c", linewidth=1.5, label="Inference Time T_comp (TorchScript CPU)")
    ax4.axhline(500.0, color="red", linestyle="--", linewidth=1.2, label="500 ms Real-Time Step Budget")
    mean_lat = np.mean(latencies)
    ax4.axhline(mean_lat, color="#005a32", linestyle=":", linewidth=1.2, label=f"Mean Latency ({mean_lat:.1f} ms)")
    ax4.set_ylabel("Latency (ms)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Time (seconds)", fontsize=12, fontweight="bold")
    ax4.set_ylim(0, 25.0) # Zoom into execution time range
    ax4.legend(loc="upper right", frameon=True, framealpha=0.9)
    ax4.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_path = Path(output_png)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\n[PLOT SAVED] Successfully generated publication figure at: {out_path.resolve()}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Real-Time Streaming BCI Telemetry")
    parser.add_argument("--subject", type=str, default="S1_data_preproc", help="DTU Subject name")
    parser.add_argument("--montage", type=str, default="near_ear_expanded", help="Electrode montage")
    parser.add_argument("--trial_idx", type=int, default=0, help="Trial index to visualize")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to model checkpoint")
    parser.add_argument("--output_png", type=str, default="/kaggle/working/streaming_bci_telemetry.png", help="Output PNG path")
    args = parser.parse_args()
    
    generate_telemetry_plot(
        subject=args.subject,
        montage=args.montage,
        trial_idx=args.trial_idx,
        checkpoint=args.checkpoint,
        output_png=args.output_png
    )
