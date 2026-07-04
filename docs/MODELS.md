# MASTER MODELS — Complete Architecture Inventory

> Every model ever implemented, with exact specifications, results, and status.

---

## Model 1: ContrastiveMatchNet (DTU Production)

**File**: [matchnet.py](file:///C:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/models/matchnet.py)
**Training Script**: [train_matchnet_loso.py](file:///C:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/training/train_matchnet_loso.py)
**Total Parameters**: 50,928
**Status**: FROZEN — production model for DTU confidence pipeline

### EEG Encoder (Modified EEGNet): 2,320 params

| # | Layer | Shape | Params | Purpose |
|---|-------|-------|--------|---------|
| 1a | Conv2d(1, 8, (1,64), pad=(0,32)) | [8,1,1,64] | 512 | Temporal bandpass filtering (1s kernel) |
| 1b | BatchNorm2d(8) | [8]+[8] | 16 | Normalize |
| 1c | Conv2d(8, 16, (8,1), groups=8) | [16,1,8,1] | 128 | Depthwise spatial filtering |
| 1d | BatchNorm2d(16) | [16]+[16] | 32 | Normalize |
| 1e | GELU + Dropout(0.25) | — | 0 | — |
| 2a | Conv2d(16, 16, (1,16), pad=(0,8), groups=16) | [16,1,1,16] | 256 | Depthwise temporal refinement |
| 2b | Conv2d(16, 16, (1,1)) | [16,16,1,1] | 256 | Pointwise channel mixing |
| 2c | BatchNorm2d(16) + GELU + Dropout(0.25) | — | 32 | — |
| 3 | Conv1d(16, 64, k=1) | [64,16,1]+[64] | 1,088 | Projection to 64-D latent |

**Receptive Field**: ~80 samples ≈ 1.25 seconds

### Audio Encoder (1D-CNN): 48,608 params

| # | Layer | Shape | Params | Purpose |
|---|-------|-------|--------|---------|
| 1 | Conv1d(28, 32, k=15, pad=7) + BN + GELU + Dropout(0.2) | [32,28,15]+[32] | 13,472+64 | Low-level spectro-temporal |
| 2 | Conv1d(32, 64, k=15, pad=7) + BN + GELU + Dropout(0.2) | [64,32,15]+[64] | 30,784+128 | Mid-level modulations |
| 3 | Conv1d(64, 64, k=1) | [64,64,1]+[64] | 4,160 | Pointwise projection |

**Kernel k=15**: ~234ms at 64Hz (syllable-level sensitivity)

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 1e-3 |
| Weight Decay | 1e-4 |
| Batch Size | 128 |
| Max Epochs | 100 (early stopped) |
| Early Stopping Patience | 10 |
| Contrastive Margin | 0.1 |
| Latent Dimension | 64 |
| Mixed Precision | CUDA AMP |
| Training Windows | 5s (320 samples), 2s hop |
| Evaluation Windows | 10s (640 samples), non-overlapping |
| Loss | Margin-based contrastive (cosine sim) |
| Evaluation Metric | Pearson correlation |

### Results

| Metric | Value |
|--------|-------|
| DTU Window Accuracy (10s, LOSO) | 69.02% (CI: 67.76%–70.28%) |
| Selective Accuracy @ 70% Coverage | 81.55% |

---

## Model 2: AAD-Conformer (KUL Production)

**File**: [aad_conformer.py](file:///C:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/models/aad_conformer.py)
**Training Script**: [train_conformer_loso.py](file:///C:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/training/train_conformer_loso.py)
**Total Parameters**: ~2,083,000
**Status**: FROZEN — production model for KUL streaming pipeline

### Architecture Summary

```
Input EEG (B, 8, T) 
    → EEGNet Stem (spatial-temporal decomposition)
    → Strided Tokenization
    → Conformer Blocks (multi-head self-attention + feedforward)
    → Regression Head
    → z_eeg (B, 64, T)
```

### Confidence Head (Late-Fusion MLP)

| Input Feature | Dimension |
|--------------|-----------|
| z_pool | 64 |
| margin | 1 |
| corr_a | 1 |
| corr_b | 1 |
| latent_norm | 1 |

**Training**: BCE loss + Outlier Exposure (random/zero EEG with target=0)

### Results

| Metric | Value |
|--------|-------|
| KUL LOSO Trial Accuracy (single seed) | 71.88% |
| KUL LOSO Trial Accuracy (5-seed mean) | 77.12% ± 9.99% |
| KUL Window Accuracy | 57.69% |
| Confidence AUROC | 0.7337 |
| Confidence ECE | 0.0998 |
| OOD (Random EEG) | 0.134 mean confidence |
| OOD (Zero EEG) | 0.139 mean confidence |

### Multi-Seed Summary
- Seed 1: 71.88% | Seed 7: 79.38% | Seed 21: 78.13% | Seed 42: 75.63% | Seed 123: 80.63%
- Paired t-test vs Ridge: p = 7.65×10⁻⁶, Cohen's d = 1.6642

---

## Model 3: EEGNet (Standalone Encoder)

**File**: [eegnet.py](file:///C:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/models/eegnet.py) (48 lines)
**Parameters**: 2,320 (standalone), 1,232 without projection head
**Status**: ACTIVE — used as encoder backbone in both MatchNet and Conformer

### Design Principles
- Depthwise-separable convolutions minimize parameters
- Temporal → Spatial decomposition mimics ICA/CSP pipeline
- F1=8 temporal filters, D=2 depth multiplier, F2=16 output features

---

## Model 4: AudioEncoder

**File**: [matchnet.py](file:///C:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/models/matchnet.py), Lines 8–29
**Parameters**: 48,608
**Status**: ACTIVE — shared-weight encoder for both audio streams in MatchNet

### Key Design Decision
Weight sharing between attended/unattended audio processing ensures symmetric encoding — the similarity comparison is meaningful only if both representations come from the same function.

---

## Model 5: TemporalCNN (ABANDONED)

**File**: [temporal_cnn.py](file:///C:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/models/temporal_cnn.py)
**Parameters**: ~69,000
**Status**: ABANDONED — failed LOSO evaluation

### Architecture
| Component | Details |
|-----------|---------|
| Stem | Conv1d(2→32, k=5) → Conv1d(32→64, k=1) |
| Multi-resolution | Parallel Conv1d k={3, 7, 15} → concat → project |
| Residual | 2× ResidualTemporalBlock (dilations 2, 4) |
| Head | Conv1d(64→1, k=1) |
| Loss | Negative Pearson correlation |

### Results
| Evaluation | Accuracy |
|-----------|----------|
| Within-subject | ~70%+ |
| LOSO | 50–55% |
| Shuffled labels | 45.83% |

**Why it failed**: Reconstruction objective + subject memorization.

---

## Model 6: ATCNet (NOT SELECTED)

**File**: [atcnet.py](file:///C:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/models/atcnet.py)
**Parameters**: ~15,000
**Status**: IMPLEMENTED but not selected for production

**Why not selected**: No meaningful LOSO accuracy gain over EEGNet (2,320 params), 6× larger, attention mechanism overfits to temporal patterns of training subjects.

---

## Model 7: EEGNet-TCN (ABANDONED)

**File**: [eegnet_tcn.py](file:///C:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/models/eegnet_tcn.py)
**Status**: ABANDONED — same LOSO failure as TemporalCNN (~50–55%)

---

## Model 8: VLAAI-Lite (ABANDONED)

**File**: [vlaai_lite.py](file:///C:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/models/vlaai_lite.py)
**Status**: ABANDONED — same LOSO failure as TemporalCNN (~50–55%)

---

## Model 9: XGBoost Confidence Classifier

**Type**: Gradient Boosted Trees (not neural)
**Parameters**: 100 trees, max depth 3
**Input Features**: 5 (margin, sim_chosen, sim_unchosen, rolling_std_margin, trial_consistency)
**Status**: ACTIVE — used in DTU confidence pipeline

### Results
| Metric | Value |
|--------|-------|
| AUROC | 0.8057 (CI: 0.7936–0.8182) |
| Margin-Only AUROC | 0.6601 |

### SHAP Feature Importance
| Feature | Weight |
|---------|--------|
| margin | 0.42 |
| rolling_std_margin | 0.35 |
| sim_chosen | 0.12 |
| trial_consistency | 0.08 |
| sim_unchosen | 0.03 |

---

## Model 10: Ridge Regression Baseline

**File**: [ridge_aad.py](file:///C:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/baselines/ridge_aad.py)
**Type**: Linear (not neural)
**Parameters**: (8 channels × 16 lags) × 1 output = 128 weights + bias
**Status**: ACTIVE — reference baseline

### Results
| Configuration | Trial Accuracy |
|--------------|---------------|
| 2ch, lags=48, λ=1.0, 10s | 55.19% |
| 8ch, lags=48, λ=1.0, 10s | ~65–69% |
| 8ch, 50s windows | ~69% |
| 8ch, trial-level (majority vote) | ~78% |

---

## Architecture Comparison Summary

| Model | Params | LOSO Acc | Status | Purpose |
|-------|--------|----------|--------|---------|
| Ridge Regression | 128 | 55–69% | Baseline | Linear floor |
| TemporalCNN | 69,000 | 50–55% | ABANDONED | Failed reconstruction |
| VLAAI-Lite | — | 50–55% | ABANDONED | Failed reconstruction |
| EEGNet-TCN | — | 50–55% | ABANDONED | Failed reconstruction |
| ATCNet | 15,000 | ≈ EEGNet | Available | Not selected (overfits) |
| EEGNet | 2,320 | — | Active | Encoder backbone |
| ContrastiveMatchNet | 50,928 | 69% | FROZEN | DTU production |
| AAD-Conformer | ~2,083,000 | 77% | FROZEN | KUL production |
| XGBoost (Confidence) | ~100 trees | AUROC 0.81 | Active | Confidence estimation |
