# MASTER PROJECT STATE — Canonical Repository Report

> Single source of truth for the EEG AAD project.

---

## 1. Mission Statement

Build a **confidence-aware selective Auditory Attention Decoding (AAD)** system for next-generation neuro-steered hearing aids. The system must decode which speaker a listener is attending to from 8-channel wearable EEG, estimate its own reliability, and abstain from prediction when uncertain — maintaining the hearing aid's current acoustic lock rather than risking a catastrophic wrong switch.

---

## 2. Repository Structure

```
EEG_8_Channel_Pipeline/
├── models/                  # 15 neural architectures
│   ├── matchnet.py          # ContrastiveMatchNet (50,928 params) — DTU production
│   ├── eegnet.py            # EEGNet (2,320 params) — encoder backbone
│   ├── aad_conformer.py     # AAD-Conformer (~2M params) — KUL production
│   ├── atcnet.py            # ATCNet (alternative backbone)
│   ├── eegnet_tcn.py        # EEGNet-TCN hybrid
│   ├── temporal_cnn.py      # TemporalCNN (~69k params, failed)
│   └── vlaai_lite.py        # VLAAI-Lite (alternative)
├── training/                # 42 training scripts
│   ├── train_matchnet_loso.py      # DTU LOSO training
│   ├── train_conformer_loso.py     # KUL LOSO training
│   ├── export_subject_distance.py  # Confidence CSV export
│   └── evidential_loss.py          # Evidential DL loss
├── decision_engine/         # 3 decision components
│   ├── context_aware_engine.py     # ContextAwarePolicyEngine (FSM)
│   └── window_buffer.py           # SequentialWindowBuffer + TemporalEvidenceAccumulator
├── decision_policy_engine.py       # DecisionPolicyEngine (original FSM)
├── src/confidence/          # Confidence estimation modules
├── analysis/                # 179 analysis/audit scripts
├── baselines/               # Ridge regression baselines
├── scenarios/               # 5 JSON scenario definitions
├── docs/                    # Research documentation
├── results/                 # Phase-organized result artifacts
├── conformer_loso_results/  # Multi-seed conformer outputs
└── checkpoints/             # Trained model weights
```

---

## 3. Datasets

### 3.1 DTU (Primary Development Dataset)
- **Source**: Technical University of Denmark (Fuglsang et al., 2017)
- **Subjects**: 18 (S1–S18), normal hearing
- **Trials**: 60 per subject
- **Trial Duration**: ~50 seconds
- **EEG**: 66 channels (64 BioSemi + 2 EXG), 64 Hz after preprocessing
- **Audio**: Danish audiobooks, dichotic presentation
- **Used Channels**: 8 peripheral (Fp1[13], Fp2[46], F7[43], F8[23], T7[50], T8[0], P7[52], P8[14])
- **Audio Representation**: 28-band Gammatone envelopes (ERB-spaced, 50–8000 Hz), ^0.3 compression
- **Label Convention**: Label 1/2 = speaker gender, NOT stream A/B. `wavA` is ALWAYS attended.
- **Known Limitation**: 100% stimulus overlap between train/test (same stories heard by all subjects)

### 3.2 KUL (Cross-Dataset Validation)
- **Source**: KU Leuven
- **Subjects**: 16 (S1–S16)
- **Trials**: 20 per subject
- **Trial Duration**: ~389 seconds
- **EEG**: 64 channels (BioSemi64), 128 Hz raw → downsampled to 64 Hz
- **Audio**: Raw `.wav` → reconstructed 28-band Gammatone to match DTU format
- **Label Convention**: `attended_ear` field + `stimuli` array determines attended stream
- **Known Limitation**: Track number NOT fixed to ear — tracks swap across trials

---

## 4. Model Inventory

### 4.1 ContrastiveMatchNet (DTU Production Model)
- **File**: [matchnet.py](file:///C:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/models/matchnet.py)
- **Total Parameters**: 50,928
- **EEG Encoder**: Modified EEGNet (2,320 params) — depthwise spatial-temporal convolutions
- **Audio Encoder**: 1D-CNN (48,608 params) — 3 cascading Conv1d layers
- **Input**: EEG [B, 8, T], Audio [B, 28, T]
- **Output**: z_eeg, z_a, z_b ∈ [B, 64, T]
- **Loss**: Margin-based contrastive (margin=0.1) using cosine similarity
- **Evaluation**: Pearson correlation between z_eeg and z_audio
- **Status**: FROZEN — used for DTU confidence research pipeline

### 4.2 AAD-Conformer (KUL Production Model)
- **File**: [aad_conformer.py](file:///C:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/models/aad_conformer.py)
- **Total Parameters**: ~2,083,000
- **Architecture**: EEGNet stem → Strided tokenization → Multi-head self-attention → Regression head
- **With Confidence Head**: Late-Fusion MLP (z_pool + margin + corr_a + corr_b + latent_norm)
- **Training**: KUL dataset, InfoNCE loss, strict LOSO
- **Status**: FROZEN — used for KUL streaming pipeline

### 4.3 Failed/Baseline Architectures
| Architecture | Params | LOSO Result | Status |
|-------------|--------|-------------|--------|
| TemporalCNN | ~69,000 | 50–55% | ABANDONED |
| VLAAI-Lite | — | ~50–55% | ABANDONED |
| EEGNet-TCN | — | ~50–55% | ABANDONED |
| ATCNet | ~15,000 | ≈ EEGNet | NOT SELECTED (no gain, more params) |

---

## 5. Confidence Systems

### 5.1 XGBoost Geometric Confidence (MatchNet/DTU)
- **Features**: margin, sim_chosen, sim_unchosen, rolling_std_margin, trial_consistency
- **Model**: XGBoost (100 trees, depth 3)
- **AUROC**: 0.8057 (CI: 0.7936–0.8182)
- **Selective Gain**: 81.55% @ 70% coverage (+12.53% over baseline)

### 5.2 Learned Confidence Head (Conformer/KUL)
- **Architecture**: Late-Fusion MLP
- **Input**: z_pool(64) + margin(1) + corr_a(1) + corr_b(1) + latent_norm(1)
- **Training**: BCE loss + Outlier Exposure (random/zero EEG)
- **AUROC**: 0.7337 (CI: 0.7303–0.7374)
- **ECE**: 0.0998
- **OOD**: Random → 0.134, Zero → 0.139

### 5.3 Known Confidence Limitations
- Dynamic range compressed [0.35, 0.52]
- 46.8% dead neurons in layer_1_ReLU
- Margin shortcut: model learns sigmoid scaling over margin, not true uncertainty
- Information limit for failure prediction: AUROC ≈ 0.59

---

## 6. Decision Policy Engines

### 6.1 DecisionPolicyEngine (v1)
- **File**: [decision_policy_engine.py](file:///C:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/decision_policy_engine.py)
- **States**: INITIALIZING → WAITING → LOCKED → SWITCHING → UNCERTAIN
- **Evidence**: Causal Log-Likelihood Ratio accumulation
- **Config**: threshold=0.85, min_lock=5, min_switch_gap=10, min_consecutive=3, max_wait=15

### 6.2 ContextAwarePolicyEngine (v2)
- **File**: [context_aware_engine.py](file:///C:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/decision_engine/context_aware_engine.py)
- **Added States**: STABILIZING, COOLDOWN
- **5 Heuristics**: difficulty scaling, evidence growth rate, oscillation penalty, hysteresis, cooldown
- **Dynamic Threshold**: Adjusts base_threshold based on estimated scene difficulty
- **Current Phase 17 Results**: True Switches=2, False Switches=2, Coverage=92.58%

### 6.3 SequentialWindowBuffer
- **File**: [window_buffer.py](file:///C:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/decision_engine/window_buffer.py)
- **Classes**: WindowPrediction (dataclass), SequentialWindowBuffer (memory), TemporalEvidenceAccumulator (Dirichlet)
- **Purpose**: Passive temporal memory — no decision logic

---

## 7. Hardware Emulator (Phase 16)

### 7.1 ContinuousSessionGenerator
- **Architecture**: Scenario JSON → DatasetAdapter → Raw concatenation → 2s sliding window (50ms hop)
- **Output**: Chronological windows with ground truth, timestamps, scene context
- **Adapters**: KULAdapter (implemented), DTUAdapter (planned)

### 7.2 Scenarios
| # | Name | Purpose |
|---|------|---------|
| 1 | stable_conversation | Baseline stability + hysteresis |
| 2 | single_shift | Single A→B→A, switch latency isolation |
| 3 | rapid_conversation | A→B→A→B (60/30/30/30s), stress cooldown |
| 4 | mixed_difficulty | Multiple trials, implicit adaptive behavior |
| 5 | long_continuous | Multi-trial block, multi-hour session sim |

### 7.3 Transition Semantics
- **Splice Timestamp**: Exact boundary defined by scenario JSON
- **Center-Sample Convention**: Window metadata assigned by center timestamp
- **Switch Latency**: `Controller_Lock_Timestamp - Splice_Timestamp`

---

## 8. Current State (Phase 17.3)

### What Works
- AAD-Conformer decodes auditory attention at 77.12% ± 9.99% (5-seed LOSO, KUL)
- ContrastiveMatchNet decodes at 69.02% (DTU LOSO)
- Confidence framework lifts effective accuracy to >80% via selective prediction
- Cross-dataset zero-shot transfer validated (KUL → DTU)
- 10/10 negative controls PASS
- Complete streaming pipeline: EEG → Conformer → Confidence → FSM → Hearing Aid Output
- Hardware emulator reproduces 5 acoustic scenarios with 50ms resolution

### What Doesn't Work Yet
- Switch/Recovery Latency: 25.41s (too slow for responsive switching)
- Audible False Switches: 22.53/hr (significant user disruption)
- Information limit: similarity-derived confidence caps at AUROC ≈ 0.59 for failure prediction
- Confidence head dynamic range compressed — needs Evidential DL replacement
- No real dry-electrode hardware validation

### Open Scientific Questions
1. Can Evidential DL (Phase 13 design) break the AUROC 0.59 information limit?
2. Does the confidence generalize to dry-electrode EEG (higher impedance, more noise)?
3. Can switch latency be reduced below 5s while maintaining stability?
4. What is the performance floor with 2 or 4 channels instead of 8?
5. Can the system handle 3+ simultaneous speakers?
