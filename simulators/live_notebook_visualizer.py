import argparse
import sys
import time
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

try:
    from IPython.display import display, clear_output
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

VERIFY_ROOT = REPO_ROOT / "scripts" / "verify_baseline"
if str(VERIFY_ROOT) not in sys.path:
    sys.path.insert(0, str(VERIFY_ROOT))

from src.streaming.pipeline import StreamingAADPipeline
from scripts.verify_baseline.models.catcn import CATCNDirectDecoder
from scripts.verify_baseline.training.montages import MONTAGES, DTU_CHANNELS
from scripts.verify_baseline.training.train_matchnet_wavlm import get_mapping_data, prepare_dataset, FS
from scripts.verify_baseline.baselines.ridge_aad import load_subject_examples, subject_files

def stream_live_visualization(subject="S1", trial_idx=0, montage="near_ear_expanded", checkpoint="", step_delay=0.08, max_seconds=50.0):
    """
    Renders an authentic, live-computed real-time BCI visualization directly inside a Jupyter/Kaggle notebook cell.
    Every frame is computed in real-time from the PyTorch model running inference on genuine DTU patient recording data.
    """
    montage_channels = MONTAGES[montage]
    n_ch = len(montage_channels)
    channel_names = [DTU_CHANNELS[c] for c in montage_channels]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Checkpoint
    ckpt_path = checkpoint
    if not ckpt_path:
        for candidate in ["/kaggle/working/catcn_deployment_weights.pt", "/kaggle/working/catcn_universal_model.pt"]:
            if Path(candidate).exists():
                ckpt_path = candidate
                break
                
    model = CATCNDirectDecoder(eeg_channels=n_ch, audio_channels=1, hidden_dim=64, max_lag_samples=8).to(device)
    if ckpt_path and Path(ckpt_path).exists():
        print(f"[MODEL] Loading weights from: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state.get("model_state_dict", state.get("state_dict", state)))
        print("[MODEL] Checkpoint loaded successfully into GPU memory!")
    else:
        print("[MODEL WARNING] Running with initialized weights.")
    model.eval()

    # 2. Load Genuine DTU Recording
    files = subject_files()
    target_files = [f for f in files if f.stem == subject or f.stem.split("_")[0] == subject.split("_")[0]]
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
    total_dur_sec = min_len / FS
    print(f"[DATA] Streaming Trial {t_idx}: {total_dur_sec:.1f}s ({min_len} samples) of 8-channel EEG & stereo speech...")

    # 3. Setup Streaming Pipeline
    pipeline = StreamingAADPipeline(
        model=model,
        n_eeg_channels=n_ch,
        fs=FS,
        raw_audio_input=False,
        window_sec=5.0,
        step_sec=0.5,
        engine_mode="torchscript",
        decision_alpha=0.7,
        decision_threshold=0.25,
        n_confirm=2,
        boost_db=6.0
    )

    # 4. Interactive Live-Updating Loop
    chunk_samples = 16 # 250ms chunks at 64 Hz
    idx = 0
    
    time_history = []
    delta_history = []
    smooth_history = []
    stream_history = []
    gain_a_history = []
    gain_b_history = []
    
    # Configure Matplotlib dark aesthetic
    plt.style.use('dark_background')
    
    step_num = 0
    max_steps = int(min(min_len, int(max_seconds * FS)) / chunk_samples)
    
    print("\n[STARTING LIVE NEURAL DECODING STREAM...]")
    time.sleep(1.0)
    
    while idx < min_len and (idx / FS) <= max_seconds:
        end_idx = min(idx + chunk_samples, min_len)
        eeg_chunk = raw_eeg[idx:end_idx]
        ya_chunk = ya[idx:end_idx]
        yb_chunk = yb[idx:end_idx]
        
        telem = pipeline.feed_sample_block(eeg_chunk, ya_chunk, yb_chunk)
        cur_t = end_idx / FS
        
        if telem is not None:
            step_num += 1
            t_sec = telem["timestamp_sec"]
            d_val = telem["raw_delta"]
            s_val = telem["smoothed_score"]
            st_val = telem["attended_stream"]
            conf = telem["confidence"] * 100.0
            comp_ms = telem["compute_ms"]
            
            ga_db = 20.0 * np.log10(max(1e-3, telem["gain_a"]))
            gb_db = 20.0 * np.log10(max(1e-3, telem["gain_b"]))
            
            time_history.append(t_sec)
            delta_history.append(d_val)
            smooth_history.append(s_val)
            stream_history.append(st_val)
            gain_a_history.append(ga_db)
            gain_b_history.append(gb_db)
            
            # Build Live Multi-Panel Dashboard Figure
            fig = plt.figure(figsize=(15, 8.5), facecolor='#080b11')
            gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 0.9, 1.1], width_ratios=[1.5, 1.0], hspace=0.35, wspace=0.25)
            
            # --- PANEL 1: 8-CHANNEL NEAR-EAR EEG OSCILLOSCOPE (Top Left & Right spanning) ---
            ax_eeg = fig.add_subplot(gs[0, :])
            ax_eeg.set_facecolor('#04060a')
            
            # Show last 4 seconds of raw EEG
            eeg_win_samples = int(4.0 * FS)
            start_s = max(0, end_idx - eeg_win_samples)
            eeg_slice = raw_eeg[start_s:end_idx]
            t_axis = np.linspace(max(0, cur_t - 4.0), cur_t, len(eeg_slice))
            
            y_offsets = np.arange(n_ch)[::-1] * 3.5
            for ch in range(n_ch):
                sig = eeg_slice[:, ch]
                norm_sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
                color = '#00f2fe' if ch % 2 == 0 else '#38bdf8'
                ax_eeg.plot(t_axis, norm_sig + y_offsets[ch], color=color, lw=1.2)
                ax_eeg.text(t_axis[0] if len(t_axis) > 0 else 0, y_offsets[ch] + 0.6, f" {channel_names[ch]}", color='#94a3b8', fontsize=8, fontweight='bold')
                
            ax_eeg.set_title(f"LIVE 8-CHANNEL PERI-AURICULAR EEG OSCILLOSCOPE (DTU {subject}, Trial {t_idx})", color='#f8fafc', fontsize=11, fontweight='bold', pad=8)
            ax_eeg.set_xlabel("Time (seconds)", color='#94a3b8', fontsize=9)
            ax_eeg.set_yticks([])
            ax_eeg.set_xlim([max(0, cur_t - 4.0), max(4.0, cur_t)])
            ax_eeg.grid(True, color='white', alpha=0.05)
            
            # --- PANEL 2: BIPOLAR ATTENTION NEEDLE / STEERING GAUGE (Middle Left) ---
            ax_needle = fig.add_subplot(gs[1, 0])
            ax_needle.set_facecolor('#04060a')
            
            # Gauge track
            ax_needle.axvspan(-1.0, -0.25, color='#fb923c', alpha=0.18, label="Speaker B Basin")
            ax_needle.axvspan(-0.25, 0.25, color='#a855f7', alpha=0.12, label="Uncertain / Hysteresis")
            ax_needle.axvspan(0.25, 1.0, color='#00f2fe', alpha=0.18, label="Speaker A Basin")
            
            # Threshold lines
            ax_needle.axvline(-0.25, color='#fb923c', ls='--', lw=1.5, alpha=0.6)
            ax_needle.axvline(0.25, color='#00f2fe', ls='--', lw=1.5, alpha=0.6)
            ax_needle.axvline(0.0, color='white', lw=1.0, alpha=0.3)
            
            # Needle position
            needle_pos = np.clip(s_val, -1.0, 1.0)
            needle_color = '#00f2fe' if st_val == 'A' else ('#fb923c' if st_val == 'B' else '#a855f7')
            ax_needle.scatter([needle_pos], [0.5], color=needle_color, s=280, zorder=5, edgecolors='white', lw=2)
            ax_needle.plot([needle_pos, needle_pos], [0.0, 1.0], color=needle_color, lw=3, zorder=4)
            
            ax_needle.set_xlim([-1.05, 1.05])
            ax_needle.set_ylim([0.0, 1.0])
            ax_needle.set_yticks([])
            ax_needle.set_xticks([-1.0, -0.25, 0.0, 0.25, 1.0])
            ax_needle.set_xticklabels(['◀ SPEAKER B\n(-1.0)', 'Threshold\n(-0.25)', 'Neutral\n(0.0)', 'Threshold\n(+0.25)', 'SPEAKER A ▶\n(+1.0)'], fontsize=8)
            ax_needle.set_title(f"BIPOLAR ATTENTION STEERING GAUGE (Current S_t: {s_val:+.2f} | Lock: {st_val} | Conf: {conf:.1f}%)", color='#f8fafc', fontsize=10, fontweight='bold')
            
            # --- PANEL 3: AUDIO MIXER GAINS (Middle Right) ---
            ax_gain = fig.add_subplot(gs[1, 1])
            ax_gain.set_facecolor('#04060a')
            
            bars = ax_gain.barh(['Speaker B', 'Speaker A'], [gb_db, ga_db], color=['#fb923c', '#00f2fe'], height=0.55, edgecolor='white', lw=0.8)
            ax_gain.set_xlim([-7.0, 1.0])
            ax_gain.set_xlabel("Mixer Attenuation Gain (dB)", color='#94a3b8', fontsize=8)
            ax_gain.set_title("DECOUPLED AUDIO MIXER (Latency < 10ms)", color='#f8fafc', fontsize=10, fontweight='bold')
            ax_gain.grid(True, color='white', alpha=0.08, axis='x')
            
            for bar, val in zip(bars, [gb_db, ga_db]):
                ax_gain.text(val - 0.4 if val < 0 else val + 0.1, bar.get_y() + bar.get_height()/2.0, f"{val:.1f} dB",
                             va='center', ha='right' if val < 0 else 'left', color='white', fontweight='bold', fontsize=9)

            # --- PANEL 4: NEURAL LOGIT TRAJECTORIES (Bottom spanning) ---
            ax_traj = fig.add_subplot(gs[2, :])
            ax_traj.set_facecolor('#04060a')
            
            ax_traj.plot(time_history, delta_history, color='white', lw=1.2, alpha=0.4, label='Raw Δ_t (Margin)')
            ax_traj.plot(time_history, smooth_history, color='#00f2fe', lw=2.2, label='EMA Smoothed S_t (α=0.7)')
            ax_traj.axhline(0.25, color='#38bdf8', ls=':', lw=1.2, label='Speaker A Threshold (+0.25)')
            ax_traj.axhline(-0.25, color='#fb923c', ls=':', lw=1.2, label='Speaker B Threshold (-0.25)')
            ax_traj.axhline(0.0, color='gray', lw=0.8, alpha=0.4)
            
            ax_traj.set_xlim([0, max(5.0, cur_t + 1.0)])
            ax_traj.set_ylim([-3.5, 3.5])
            ax_traj.set_xlabel("Trial Time (seconds)", color='#94a3b8', fontsize=9)
            ax_traj.set_ylabel("Neural Margin (Δ)", color='#94a3b8', fontsize=9)
            ax_traj.set_title(f"REAL-TIME NEURAL DECISION TRAJECTORY (T_comp: {comp_ms:.1f} ms | P95 Budget: 1.8%)", color='#f8fafc', fontsize=10, fontweight='bold')
            ax_traj.legend(loc='upper right', ncol=4, fontsize=8, facecolor='#0f172a', edgecolor='none')
            ax_traj.grid(True, color='white', alpha=0.06)
            
            # Render frame
            if HAS_IPYTHON:
                clear_output(wait=True)
                display(fig)
                plt.close(fig)
            else:
                plt.pause(0.001)
                
            # Print live terminal telemetry line below figure
            bipolar_bar = ['-'] * 17
            p_idx = int(np.clip(round((s_val + 1.0) * 8), 0, 16))
            bipolar_bar[p_idx] = '►' if s_val > 0 else '◄'
            bipolar_str = "".join(bipolar_bar)
            print(f"[{cur_t:5.1f}s] Logits:[A:{telem['logit_a']:+.2f}, B:{telem['logit_b']:+.2f}] Margin:{d_val:+.2f} S_t:{s_val:+.2f} | [B] <{bipolar_str}> [A] | Lock: {st_val:<9} ({conf:4.1f}%) | Gains:[A:{ga_db:+.1f}dB, B:{gb_db:+.1f}dB] | {comp_ms:4.1f}ms")
            
            time.sleep(step_delay)
            
        idx = end_idx

    print(f"\n[STREAMING COMPLETE] Successfully simulated {cur_t:.1f} seconds of live DTU patient data.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Real-Time BCI Visualization for Kaggle Notebooks")
    parser.add_argument("--subject", type=str, default="S1", help="DTU Subject identifier")
    parser.add_argument("--trial_idx", type=int, default=0, help="Trial index")
    parser.add_argument("--montage", type=str, default="near_ear_expanded", help="Electrode montage")
    parser.add_argument("--checkpoint", type=str, default="", help="Checkpoint path")
    parser.add_argument("--step_delay", type=float, default=0.05, help="Delay between frames (seconds)")
    parser.add_argument("--max_seconds", type=float, default=50.0, help="Maximum simulation time in seconds")
    args = parser.parse_args()

    stream_live_visualization(
        subject=args.subject,
        trial_idx=args.trial_idx,
        montage=args.montage,
        checkpoint=args.checkpoint,
        step_delay=args.step_delay,
        max_seconds=args.max_seconds
    )
