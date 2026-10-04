# Temporal Gating & Sticky State Retention Benchmark

## Executive Summary

Following extensive Multi-Window (1s–20s) and Neural Lag (0–250ms) sweeps across the 18 DTU subjects, the **5.0-second causal window with 0.5-second control updates (2 Hz)** was frozen as the core product target for interactive hearing aid speech steering.

While earlier selective gating (Method F) achieved 77.5% selective accuracy, it suffered from **blunt HOLD behavior (39.9% to 51.6% of the conversation in neutral silence)**. To solve this, we implemented and benchmarked **5 distinct control architectures** on the frozen CA-TCN backbone.

### The Grand Benchmark Comparison

| Architecture | Selective Accuracy | False Switches / min | Useful Boost Coverage (+6 dB) | Neutral Silence (HOLD) | DSP Cost / Complexity |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1. Forced Base (Raw CA-TCN)** | 64.0% | 9.46 / min | 100.0% (unstable) | 0.0% | Baseline ($O(1)$) |
| **2. Blunt HOLD (Method F)** | 77.5% | 1.06 / min | 0.0% (neutral during drops) | **39.9%** (up to 51.6%) | Low ($O(1)$) |
| **3. Sticky State Retention** | **74.0%** | **1.06 / min** | **69.5%** | **6.0%** | **Optimal ($O(1)$, 0.5 µs)** |
| **4. Tiny GRU (Neural Gate)** | 67.7% | 2.55 / min | 70.8% | 4.8% | High (Matrix RNN) |
| **5. Sticky Pro (Asym+SPRT)** | 68.6% | 2.14 / min | 65.9% | 3.9% | Low ($O(1)$) |

---

## Analysis of Architectures

### 1. Why Forced Base Fails in Production
- **9.46 false switches/minute** means the hearing aid involuntarily flips audio steering between speakers **every 6.3 seconds**. Natural EEG noise and conversational pauses cause acoustic ping-ponging that causes severe cognitive fatigue.

### 2. Why Blunt HOLD is a "Hollow Victory"
- Blunt HOLD achieves 77.5% accuracy primarily by **shutting off directional assistance 40% to 52% of the time**. On challenging subjects like S4, the user spends 51.6% of their conversation in neutral audio. Whenever a conversational pause or low-margin interval occurs, the hearing aid drops the speaker.

### 3. Why Standard Sticky Retention (Strategy 3) Won
- **Acoustic Continuity:** Neutral HOLD collapses from **39.9% down to 6.0%**.
- **Useful Boost Coverage:** Actively amplifies the attended speaker with $+6\text{ dB}$ for **~70% of the entire conversation**.
- **Switch Stability:** False switches remain rock-solid at **1.06 / min**.
- **Zero Training Cost:** Closed-form, analytical, and executes in **0.0005 ms** on any ultra-low-power hearing aid DSP.

### 4. Why Neural (GRU) and Complex (Sticky Pro) Gates Degraded
- **Tiny GRU (67.7% Acc, 2.55 F.Sw/m):** With only 12 calibration trials (~12 minutes of EEG), recurrent weights overfit to transient baseline drift, causing boundary jitter.
- **Sticky Pro (68.6% Acc, 2.14 F.Sw/m):** By accumulating evidence on raw instantaneous margins ($\max(-s_t, -m_t)$), single-step EEG noise spikes bypassed the temporal filter and triggered premature, unintended speaker switches.

---

## Roadmap: How to Push Accuracy Beyond 74.0%

While 74.0% with 1.06 switches/min and 70% active coverage is a solid baseline, higher accuracy is necessary for a premium product. 

The gating mechanism has successfully eliminated the control-layer penalty. The remaining bottleneck is the **raw signal-to-noise ratio (SNR) of the 5.0-second neural margin itself**.

Here are the 4 concrete scientific paths to push accuracy to **80%+ at 5.0 seconds**:

### Path 1: Dual-Scale Multi-Window Fusion (Anchor-Gated Decision)
- **The Concept:** From our window sweep, we know that a 10s window achieves **82.7%** and a 20s window achieves **89.3%**.
- **The Upgrade:** Run a fast 5.0s window at 2 Hz for low latency, combined with a slow 10.0s "Anchor" window evaluated at 0.5 Hz.
- **The Rule:** If the 5.0s window suggests a speaker switch, but the 10.0s anchor disagrees, the switch is rejected. This injects the 83% accuracy of the 10s window into the 5s responsive stream.

### Path 2: Subject-Specific 1×1 Spatial Adapter (EEG Geometry Alignment)
- **The Concept:** 8 near-ear electrodes sit at slightly different skull positions on each subject, causing phase and amplitude distortion across channels.
- **The Upgrade:** Freeze the CA-TCN universal backbone, but calibrate a lightweight linear $8 \times 8$ spatial projection matrix $W_{\text{spatial}}$ on the 12 calibration trials. This aligns the subject's physical skull montage to the canonical feature space before temporal convolutions.

### Path 3: Speech Representation Upgrade (Beyond Gammatone Envelopes)
- **The Concept:** Gammatone envelopes only capture gross acoustic energy modulations; they discard phonetic, pitch, and voice identity features.
- **The Upgrade:** Feed multi-band acoustic representations or pre-trained speech features (e.g. self-supervised representations from WavLM/Whisper encoder) into the CA-TCN audio stream.

### Path 4: Calibrated Temperature-Adaptive Hysteresis
- **The Concept:** Currently, hysteresis thresholds ($\theta_{\text{switch}}, \theta_{\text{maintain}}$) are calibrated on a coarse grid.
- **The Upgrade:** Dynamically scale the hysteresis barriers inversely proportional to the calibrated temperature $T_{\text{calib}}$. Low-SNR subjects automatically receive wider hysteresis deadbands, preserving accuracy.

---

## Git Release Reference
- **Release Branch:** `release/sticky_gating_5s`
- **Tracked Remotes:** `isef` (`https://github.com/Loki3306/ISEF_Project.git`) & `origin` (`https://github.com/Loki3306/EEG_8_Channel_Pipeline.git`)
- **Key Implementation:** [`src/selective_aad/temporal_gate.py`](file:///c:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/src/selective_aad/temporal_gate.py) (`StickyHysteresisGate`, `SignalQualityMonitor`)
- **Evaluation Suite:** [`scripts/verify_baseline/audit/run_temporal_gating_benchmark.py`](file:///c:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/scripts/verify_baseline/audit/run_temporal_gating_benchmark.py)
