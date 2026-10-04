# Subject-Specific 8×8 Spatial Adapter Benchmark (Frozen 5.0s CA-TCN)

## Executive Summary & Milestone Status

This milestone experimentally evaluates **Path B: Subject-Specific $8 \times 8$ Spatial Adaptation** against the unadapted baseline and the previously frozen **Sticky Hysteresis Gate** ([`release/sticky_gating_5s`](file:///c:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New)).

Following this milestone, **all neural decoders and spatial adapters are 100% frozen** on branch `release/spatial_adapter_5s` as development transitions into the downstream acoustic layer (audio amplification and unattended talker suppression).

---

## Architecture & Scientific Controls

1. **Frozen Universal Backbone**:
   - Universal pre-trained CA-TCN direct decoder (anticausal EEG encoder with ~234 ms RF, causal audio encoder with ~984 ms RF, multi-lag cross-correlation head).
   - **100% Frozen** (`requires_grad = False`, evaluated in `eval()` mode).
2. **Subject-Specific $8 \times 8$ Spatial Adapter (`SpatialEEGAdapter`)**:
   - Strictly **64 trainable scalar parameters** ($W \in \mathbb{R}^{8 \times 8}$).
   - Initialized as Identity matrix $I_8$.
   - Trained *only* on the 12 calibration trials per subject (15 epochs, AdamW, $\text{LR} = 10^{-3}$).
   - Regularized towards identity via Frobenius norm penalty:
     $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}}(\Delta, y) + \lambda_{\text{reg}} \|W - I_8\|_F^2 \quad (\lambda_{\text{reg}} = 0.05)$$
3. **Temporal Control Layer**:
   - Downstream **Sticky Hysteresis Gate** with symmetric speech pause retention, temperature calibration, and artifact safety monitoring.

---

## Grand Summary: 18-Subject Benchmark Results (5.0s Windows, 0.5s Steps)

Evaluation conducted on unseen test trials (48 test trials per subject):

| Subject | 1. Raw Zero-Shot (5s)<br>Accuracy \| FalseSw | 2. Raw 8×8 Adapted (5s)<br>Accuracy \| $\Delta\text{Raw}$ \| FalseSw | 3. Adapted + Sticky Gate<br>Accuracy \| Useful Boost% \| FalseSw | HOLD% |
| :--- | :---: | :---: | :---: | :---: |
| **S1** | 65.6% \| 8.68/m | 64.1% \| -1.5 pp \| 8.90/m | 70.2% \| 67.5% \| 1.57/m | 3.9% |
| **S2** | 65.8% \| 8.52/m | 66.1% \| +0.3 pp \| 8.52/m | 73.4% \| 71.5% \| 1.04/m | 2.6% |
| **S3** | 59.1% \| 9.59/m | 63.7% \| **+4.6 pp** \| 8.10/m | 72.3% \| 71.0% \| 1.24/m | 1.7% |
| **S4** | 63.9% \| 9.70/m | 67.9% \| **+4.1 pp** \| 9.23/m | 77.2% \| 75.6% \| 1.26/m | 2.1% |
| **S5** | 62.6% \| 8.90/m | 61.1% \| -1.5 pp \| 9.37/m | 68.7% \| 65.2% \| 1.21/m | 5.0% |
| **S6** | 54.3% \| 10.30/m | 54.9% \| +0.6 pp \| 9.53/m | 57.3% \| 56.4% \| 1.76/m | 1.6% |
| **S7** | 74.1% \| 7.64/m | 76.4% \| +2.4 pp \| 7.17/m | **90.1%** \| **82.9%** \| **0.38/m** | 7.9% |
| **S8** | 74.2% \| 7.42/m | 76.9% \| +2.6 pp \| 7.36/m | **93.1%** \| **86.6%** \| **0.22/m** | 7.0% |
| **S9** | 61.4% \| 9.37/m | 63.7% \| +2.3 pp \| 8.98/m | 70.0% \| 66.6% \| 1.13/m | 4.9% |
| **S10** (Hardest) | 55.7% \| 9.42/m | 59.7% \| **+4.0 pp** \| 9.45/m | 70.9% \| 63.9% \| 1.07/m | 9.8% |
| **S11** (Non-responder) | 48.2% \| 9.92/m | 49.7% \| +1.5 pp \| 9.73/m | 46.0% \| 40.1% \| 1.24/m | 12.8% |
| **S12** | 63.3% \| 8.46/m | 64.5% \| +1.2 pp \| 8.74/m | 76.3% \| 71.8% \| 0.91/m | 5.9% |
| **S13** | 64.4% \| 9.12/m | 67.8% \| **+3.4 pp** \| 8.13/m | 77.4% \| 68.2% \| 0.66/m | 11.9% |
| **S14** | 63.2% \| 8.96/m | 65.4% \| +2.2 pp \| 7.80/m | 73.3% \| 70.3% \| 1.07/m | 4.1% |
| **S15** | 74.7% \| 6.68/m | 76.4% \| +1.7 pp \| 6.81/m | **86.8%** \| **84.1%** \| **0.66/m** | 3.1% |
| **S16** | 61.6% \| 9.34/m | 67.4% \| **+5.8 pp** \| 8.65/m | **81.0%** \| **73.1%** \| **0.60/m** | 9.8% |
| **S17** | 59.4% \| 9.64/m | 59.7% \| +0.2 pp \| 9.12/m | 65.6% \| 64.4% \| 1.51/m | 1.9% |
| **S18** | 65.6% \| 8.87/m | 67.4% \| +1.8 pp \| 8.76/m | 77.4% \| 75.3% \| 1.07/m | 2.7% |
| **AVERAGE** | **63.2%** \| **8.92/m** | **65.2%** \| **+2.0 pp** \| **8.58/m** | **73.7%** \| **69.7%** \| **1.03/m** | **5.3%** |

---

## Key Scientific Conclusions

1. **Re-alignment of Skull Montage Works**:
   - 16 out of 18 subjects showed positive raw accuracy gain ($\Delta\text{Raw} > 0$), with population raw accuracy increasing from **63.2% to 65.2% (+2.0 pp)** on strictly unseen trials.
   - S16 exhibited the greatest realignment benefit (**+5.8 pp raw**, reaching **81.0% gated selective accuracy** with only 0.60 false switches/min).
2. **Spatial Realignment Alone Cannot Suppress False Switches**:
   - Tier 1: 8.92 false switches/min.
   - Tier 2: 8.58 false switches/min.
   - Without temporal hysteresis, raw sliding windows chatter continuously during natural conversational speech pauses.
3. **Synergy in Tier 3 (Adapted 8×8 + Sticky Gate)**:
   - High-SNR subjects achieve clinical-grade reliability:
     - **S8: 93.1% accuracy, 86.6% useful boost, 0.22 false switches/min** (1 flip every 4.5 minutes).
     - **S7: 90.1% accuracy, 82.9% useful boost, 0.38 false switches/min** (1 flip every 2.6 minutes).
     - **S15: 86.8% accuracy, 84.1% useful boost, 0.66 false switches/min** (1 flip every 1.5 minutes).
   - Population false switches dropped from **8.92/min to 1.03/min** (**88% reduction**).
   - Useful boost coverage reached **69.7%** with an abstention (HOLD) rate of only **5.3%**.

---

## Frozen Checkpoints & Reproducibility

- Benchmark Script: `scripts/verify_baseline/audit/run_spatial_adapter_benchmark.py`
- Model Definition: `src/models/spatial_adapter.py`
- Gate Definition: `src/selective_aad/temporal_gate.py`
- Frozen Branch: `release/spatial_adapter_5s`
