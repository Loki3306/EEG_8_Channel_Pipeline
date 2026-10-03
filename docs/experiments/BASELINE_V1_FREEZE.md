# BASELINE V1.0 SPECIFICATION & EXPERIMENTAL FREEZE
**Status:** FROZEN  
**Date:** October 4, 2026  
**Scope:** 4-Subject Cross-Validation Benchmark (`S1`, `S5`, `S9`, `S15`)  
**Git Tag:** `v1.0-baseline-4subj`  
**Git Rollback Branch:** `checkpoint/baseline-v1-frozen`  

---

## 1. Executive Summary & Purpose
This document establishes the canonical **Baseline v1.0** reference suite for the Auditory Attention Decoding (AAD) research project. 

To prevent moving reference points, data leakage, and architectural drift during subsequent within-subject (LOTO) or calibration experiments, all definitions, hyperparameters, preprocessing pipelines, and empirical benchmark numbers documented here are **strictly frozen**.

---

## 2. Frozen Montage Suite Definitions (5 Conditions)

| Baseline ID | Montage Identifier | Electrode Channels (DTU 64 Standard) | Role & Operational Classification |
| :--- | :--- | :--- | :--- |
| **B0** | `standard_64` | All 64 scalp electrodes (Channels 0–63) | **Laboratory Reference Upper Bound** |
| **B1** | `near_ear_expanded` | `T7`, `T8`, `TP7`, `TP8`, `CP5`, `CP6`, `FC5`, `FC6` | **Hardware-Constrained Wearable Baseline** |
| **B2** | `central` | `C1`, `C2`, `C3`, `C4`, `C5`, `C6`, `CZ`, `FCZ` | **Sensorimotor / Vertex Anatomical Prior** |
| **B3** | `best8_correlation` | Dynamically selected per training fold ($g(\mathcal{D}_{\text{train}})$) | **Data-Driven Correlation Heuristic Baseline** |
| **B4** | `random_8_seed1` | `F3`, `TP7`, `IZ`, `FPZ`, `F6`, `CZ`, `CP6`, `PO8` (Seed 42) | **Unstructured Chance Random Control** |

> [!IMPORTANT]
> **Strict Semantic Boundary on B3 (`best8_correlation`):**
> This montage is strictly a greedy envelope cross-correlation heuristic selected on the training fold without test leakage. It must never be described as the "globally optimal 8 channels" ($\binom{64}{8} \approx 4.4 \times 10^9$ possibilities).

---

## 3. Validated Benchmark Performance (4-Subject LOSO Suite)

Evaluated across held-out subjects `S1_data_preproc`, `S5_data_preproc`, `S9_data_preproc`, and `S15_data_preproc` under identical training seeds, preprocessing, and model architectures.

### Aggregate Performance (Mean ± SD)
| Baseline ID | Montage | Channels | 5s Window | 10s Window | 20s Window | **40s Window** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **B0** | `standard_64` | 64 | 67.3 ± 3.1% | 72.6 ± 3.6% | 80.4 ± 4.7% | **86.2 ± 4.8%** |
| **B1** | `near_ear_expanded` | 8 | 62.1 ± 4.3% | 68.7 ± 5.7% | 74.4 ± 8.0% | **83.3 ± 12.1%** |
| **B2** | `central` | 8 | 65.0 ± 2.8% | 71.2 ± 2.7% | 76.2 ± 1.8% | **82.1 ± 4.9%** |
| **B3** | `best8_correlation` | 8 | 61.3 ± 0.6% | 65.2 ± 0.6% | 70.4 ± 5.5% | **80.0 ± 6.2%** |
| **B4** | `random_8_seed1` | 8 | 57.9 ± 2.4% | 60.7 ± 1.3% | 63.3 ± 1.8% | **68.3 ± 3.1%** |

### Per-Subject 40-Second Accuracy & Delta Breakdown
$$\Delta_{\text{Near-Ear}} = \text{Acc}_{64} - \text{Acc}_{\text{Near-Ear}}$$
$$\text{Gain}_{\text{Near vs Corr}} = \text{Acc}_{\text{Near-Ear}} - \text{Acc}_{\text{Correlation-8}}$$

| Subject | 64ch Ref (B0) | Near-Ear 8ch (B1) | Corr-Selected 8 (B3) | Random-8 (B4) | $\Delta_{\text{Near-Ear}}$ | Gain vs Corr |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **S1** | 90.0% | 90.0% | 83.3% | 68.3% | **+0.0 pp** | **+6.7 pp** |
| **S5** | 86.7% | 85.0% | 70.0% | 63.3% | **+1.7 pp** | **+15.0 pp** |
| **S9** | 78.3% | 63.3% | 80.0% | 70.0% | **+15.0 pp** | **-16.7 pp** |
| **S15** | 90.0% | 95.0% | 86.7% | 71.7% | **-5.0 pp** | **+8.3 pp** |
| **MEAN** | **86.2%** | **83.3%** | **80.0%** | **68.3%** | **+2.9 pp** | **+3.3 pp** |

---

## 4. Channel Selection Stability (Correlation Heuristic across Folds)
In the 4 folds of cross-subject cross-validation, the correlation selector consistently targeted the following electrodes:

* **$P9$ (Parietal):** 4/4 folds (100.0%)
* **$O1$ (Occipital / Visual):** 4/4 folds (100.0%)
* **$P7$ (Parietal):** 4/4 folds (100.0%)
* **$PO7$ (Parieto-Occipital):** 3/4 folds (75.0%)
* **$IZ$ (Inion / Occipital):** 3/4 folds (75.0%)
* **$C1, FC3, FC6, FT8, OZ$:** 2/4 folds (50.0%)
* **$C3, CZ, F6, O2$:** 1/4 folds (25.0%)

> [!NOTE]
> This demonstrates reproducible posterior-channel preference under this selection rule across independent 17-subject training subsets.

---

## 5. Frozen Architectural & Training Hyperparameters

All code in this baseline is locked to the following specification:

### Model: CA-TCN Direct AAD Decoder
* **Audio Encoder:** Strictly causal depthwise-separable TCN (5 layers, dilations $d \in [1, 2, 4, 8, 16]$, kernel size $K=3$, receptive field $\approx 984.4\text{ ms}$).
* **EEG Encoder:** Strictly anticausal depthwise-separable TCN (3 layers, dilations $d \in [1, 2, 4]$, kernel size $K=3$, receptive field $\approx 234.4\text{ ms}$) preceded by a 1D spatial mixing projection ($C \rightarrow 64$) and `BatchNorm1d(64)`.
* **Classifier Head:** Multi-lag cross-correlation across $\tau \in [-8, +8]$ samples ($\pm 125\text{ ms}$ at 64 Hz) $\rightarrow$ linear classifier enforcing anti-symmetric logit difference $\Delta = \text{logit}_A - \text{logit}_B$.
* **Hidden Dimension:** 64
* **Dropout:** 0.2

### Signal Preprocessing & Bandpass
* **Sampling Rate:** $f_s = 64\text{ Hz}$
* **EEG Filter:** Bidirectional 2nd-order zero-phase Butterworth bandpass, **$1.0\text{ Hz} - 6.0\text{ Hz}$**.
* **Audio Filter:** Gammatone 28-band envelope averaged across bands to a 1D broadband envelope.
* **Normalization:** Per-channel mean-centering and standard deviation scaling ($z$-score).

### Training Configuration
* **Optimizer:** Adam ($\text{lr} = 2 \times 10^{-4}$, $\text{weight\_decay} = 10^{-4}$).
* **LR Scheduler:** CosineAnnealingLR ($T_{\text{max}} = 15$, $\eta_{\text{min}} = 10^{-6}$).
* **Loss Function:** Binary Cross-Entropy with Logits using symmetric audio-stream chunk swapping.
* **Precision:** Mixed Precision (`torch.amp.autocast('cuda')`).
* **Training Window:** 5.0 seconds (320 samples).
* **Training Hop:** 2.5 seconds (160 samples, 50% overlap).
* **Batch Size:** 256
* **Epochs:** 15 (Early stopping: $\text{min\_epochs} = 8$, $\text{patience} = 8$).

---

## 6. Rollback Instructions
To restore the repository state exactly to this frozen baseline at any point in the future:

```bash
# Checkout the frozen baseline branch
git checkout checkpoint/baseline-v1-frozen

# Or checkout the tagged release directly
git checkout tags/v1.0-baseline-4subj
```
