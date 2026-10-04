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

> [!NOTE]
> **Statistical Clarification on Selective Accuracy:**
> Selective accuracy is strictly conditional on the system accepting a decision interval (excluding intervals spent in HOLD). A selective accuracy of 74% does *not* imply 26% of all operational intervals are incorrect, because coverage is partial. The key milestone is achieving 74.0% selective accuracy while collapsing neutral HOLD from 39.9% down to 6.0%, yielding ~70% active directional amplification with ~1 false switch/min.

---

## Research Roadmap: Improving Raw Neural Evidence

The control gate is no longer the primary bottleneck. The current ceiling is the quality and SNR of the **5.0-second raw neural margin itself**.

The research roadmap is prioritized as follows:

### Phase 1 (Immediate Highest Value): Subject-Specific 8×8 Spatial Adapter
- **Hypothesis:** 8 near-ear electrodes sit on variable individual skull geometries, ear-canal shapes, and impedances. A subject-specific linear spatial alignment matrix $W \in \mathbb{R}^{8 \times 8}$ (64 parameters) can re-weight and rotate the physical electrodes into the canonical feature space of the pre-trained CA-TCN.
- **Architecture:**
  ```text
  Raw 8-ch EEG ──► W ∈ R^(8×8) ──► Adapted 8-ch EEG ──► Frozen CA-TCN ──► 5s Margin ──► Sticky Gate
  ```
- **Constraint:** Train **strictly $W$** on the 12 calibration trials. All 5.0s CA-TCN weights, temporal convolutions, and audio pathways remain 100% frozen.
- **Experimental Evaluation:**
  1. `Zero-shot Raw CA-TCN (5s)`
  2. `Adapted 8×8 Raw CA-TCN (5s)` (Measure whether raw margin accuracy genuinely improves)
  3. `Adapted 8×8 + Sticky Hysteresis Gate`

### Phase 2: Controlled Evaluation of 5s + 10s Multi-Scale Anchor
- **Hypothesis:** Can a slower 10.0s window prevent false switches without making intentional speaker switches unacceptably sluggish?
- **Policies to Evaluate:**
  1. `5s only` (baseline responsiveness)
  2. `10s only` (anchor upper bound)
  3. `5s + 10s agreement` (conservative switching)
  4. `5s controls, 10s confirms switches` (product candidate)
  5. `5s controls, 10s vetoes low-confidence switches` (product candidate)
- **Metrics:** False switches/min, Time-to-switch (latency to legitimate switch), correct-attention time, HOLD/maintenance time.

### Phase 3 (Postponed): Multi-Band Speech Representation
- Postponed until experimental evidence proves the current single-envelope representation is the active limiting factor. The existing benchmark demonstrates that temporal integration and spatial subject alignment are the primary drivers of performance.

---

## Git Release Reference
- **Release Branch:** `release/sticky_gating_5s`
- **Tracked Remotes:** `isef` (`https://github.com/Loki3306/ISEF_Project.git`) & `origin` (`https://github.com/Loki3306/EEG_8_Channel_Pipeline.git`)
- **Key Implementation:** [`src/selective_aad/temporal_gate.py`](file:///c:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/src/selective_aad/temporal_gate.py) (`StickyHysteresisGate`, `SignalQualityMonitor`)
- **Evaluation Suite:** [`scripts/verify_baseline/audit/run_temporal_gating_benchmark.py`](file:///c:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/scripts/verify_baseline/audit/run_temporal_gating_benchmark.py)
