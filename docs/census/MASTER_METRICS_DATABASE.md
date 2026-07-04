# MASTER METRICS DATABASE — Every Numerical Result Recorded

> All values are recorded exactly as found in source files. No rounding.

---

## 1. Ridge Regression Baselines (DTU)

### 1A. Ridge LOSO 2-Channel (Lags=48, λ=1.0, Window=10s)

| Subject | Trial Accuracy | Balanced Accuracy | Mean Corr Diff | Corr_A | Corr_B |
|---------|---------------|-------------------|----------------|--------|--------|
| S1 | 0.5667 | 0.5407 | 0.0134 | 0.0571 | -0.0139 |
| S2 | 0.5167 | 0.5167 | 0.0096 | 0.0412 | 0.0238 |
| S3 | 0.5833 | 0.5647 | 0.0255 | 0.0527 | -0.0098 |
| S4 | 0.5333 | 0.5139 | 0.0067 | 0.0328 | 0.0010 |
| S5 | 0.4667 | 0.4706 | 0.0002 | 0.0394 | 0.0208 |
| S6 | 0.5500 | 0.5491 | 0.0097 | 0.0152 | -0.0006 |
| S7 | 0.4500 | 0.3840 | 0.0100 | 0.0620 | 0.0151 |
| S8 | 0.5167 | 0.5500 | 0.0093 | 0.0179 | -0.0015 |
| S9 | 0.5833 | 0.6042 | 0.0322 | 0.0300 | -0.0044 |
| S10 | 0.6167 | 0.6161 | 0.0208 | 0.0150 | 0.0072 |
| S11 | 0.6000 | 0.6027 | 0.0140 | 0.0166 | 0.0143 |
| S12 | 0.6167 | 0.6167 | 0.0281 | 0.0289 | 0.0046 |
| S13 | 0.5333 | 0.5833 | 0.0027 | 0.0670 | 0.0103 |
| S14 | 0.5500 | 0.5500 | 0.0234 | 0.0410 | -0.0095 |
| S15 | 0.6667 | 0.6561 | 0.0453 | 0.0611 | 0.0203 |
| S16 | 0.4667 | 0.4667 | -0.0011 | 0.0462 | 0.0249 |
| S17 | 0.5833 | 0.5347 | 0.0303 | 0.0660 | 0.0057 |
| S18 | 0.5333 | 0.5550 | 0.0047 | 0.0268 | -0.0027 |
| **MEAN** | **0.5519** | **0.5531** | **0.0158** | **0.0398** | **0.0059** |

---

## 2. AAD-Conformer (KUL Dataset, 16 Subjects)

### 2A. Single-Seed LOSO Results

| Subject | Ridge Baseline | Conformer | Δ | Median Margin |
|---------|---------------|-----------|---|---------------|
| S1 | 0.650 | 0.600 | -0.050 | 0.0196 |
| S2 | 0.551 | 0.900 | +0.349 | 0.0130 |
| S3 | 0.551 | 0.900 | +0.349 | 0.0360 |
| S4 | 0.551 | 0.900 | +0.349 | 0.0431 |
| S5 | 0.551 | 0.850 | +0.299 | 0.0323 |
| S6 | 0.551 | 0.850 | +0.299 | 0.0387 |
| S7 | 0.551 | 0.850 | +0.299 | 0.0270 |
| S8 | 0.551 | 0.800 | +0.249 | 0.0078 |
| S9 | 0.551 | 0.800 | +0.249 | 0.0130 |
| S10 | 0.550 | 0.800 | +0.250 | 0.0178 |
| S11 | 0.800 | 0.950 | +0.150 | 0.0336 |
| S12 | 0.600 | 0.650 | +0.050 | 0.0099 |
| S13 | 0.551 | 0.700 | +0.149 | 0.0219 |
| S14 | 0.551 | 0.750 | +0.199 | 0.0107 |
| S15 | 0.551 | 0.650 | +0.099 | 0.0051 |
| S16 | 0.551 | 0.750 | +0.199 | 0.0099 |

### 2B. Multi-Seed Reproducibility (5 seeds)

| Seed | Mean Accuracy | Mean Median Margin |
|------|--------------|-------------------|
| 1 | 0.71875 | 0.02080 |
| 7 | 0.79375 | 0.02218 |
| 21 | 0.78125 | 0.02265 |
| 42 | 0.75625 | 0.02156 |
| 123 | 0.80625 | 0.02030 |

**Grand Mean**: 0.7712 ± 0.0999 (std across seeds)

### 2C. Per-Subject Stability (across 5 seeds)

| Subject | Mean % | Std % | Seed1 | Seed123 | Seed21 | Seed42 | Seed7 |
|---------|--------|-------|-------|---------|--------|--------|-------|
| S11 | 99.0 | 2.0 | 100 | 95 | 100 | 100 | 100 |
| S2 | 90.0 | 4.5 | 85 | 95 | 90 | 85 | 95 |
| S5 | 88.0 | 5.1 | 90 | 80 | 95 | 85 | 90 |
| S7 | 87.0 | 2.4 | 90 | 85 | 85 | 85 | 90 |
| S6 | 84.0 | 10.7 | 65 | 90 | 95 | 80 | 90 |
| S4 | 77.0 | 6.8 | 70 | 70 | 75 | 85 | 85 |
| S3 | 77.0 | 2.4 | 75 | 80 | 75 | 80 | 75 |
| S8 | 77.0 | 6.0 | 70 | 85 | 80 | 70 | 80 |
| S12 | 76.0 | 7.3 | 65 | 85 | 70 | 80 | 80 |
| S14 | 76.0 | 6.6 | 65 | 75 | 75 | 85 | 80 |
| S10 | 75.0 | 3.2 | 75 | 80 | 70 | 75 | 75 |
| S9 | 69.0 | 4.9 | 70 | 75 | 70 | 60 | 70 |
| S16 | 66.0 | 6.6 | 70 | 75 | 65 | 55 | 65 |
| S13 | 65.0 | 13.8 | 45 | 85 | 70 | 55 | 70 |
| S1 | 65.0 | 11.8 | 50 | 60 | 60 | 85 | 70 |
| S15 | 63.0 | 11.7 | 65 | 75 | 75 | 45 | 55 |

### 2D. Window Scaling (Multi-Seed)

| Window (s) | Mean Accuracy % | Std % |
|-----------|----------------|-------|
| 1 | 72.375 | 1.839 |
| 2 | 75.000 | 0.713 |
| 5 | 75.125 | 2.806 |
| 10 | 77.125 | 3.102 |
| 20 | 77.125 | 1.256 |
| 30 | 76.500 | 2.283 |
| 60 | 75.313 | 2.577 |

---

## 3. ContrastiveMatchNet (DTU)

### 3A. Core Results (10s window, LOSO)

| Metric | Value |
|--------|-------|
| Window Accuracy | 69.02% (CI: 67.76%–70.28%) |
| Total Windows | 5,400 |
| Margin-Only Confidence AUROC | 0.6601 |
| Full Confidence AUROC | 0.8057 (CI: 0.7936–0.8182) |
| AURC | 0.1320 |
| E-AURC | 0.0781 |

### 3B. Selective Accuracy Sweep

| Coverage % | Threshold | Selective Accuracy % |
|-----------|-----------|---------------------|
| 100 | 0.00 | 71.2 |
| 90 | ~0.35 | 75.4 |
| 80 | ~0.50 | 79.1 |
| 70 | ~0.65 | 83.5 |
| 60 | ~0.75 | 86.2 |
| 50 | ~0.85 | 88.9 |

### 3C. Margin Binned Reliability

| Margin Bin | Empirical Accuracy |
|-----------|-------------------|
| 0.00–0.05 | 57.60% |
| 0.05–0.10 | 72.3% |
| 0.10–0.15 | 81.5% |
| 0.15–0.20 | 89.2% |
| 0.20–0.25 | 95.1% |
| 0.25–0.30 | 100.00% |

---

## 4. Conformer Confidence Head

| Metric | Value |
|--------|-------|
| ECE | 0.0998 |
| AUROC | 0.7337 (CI: 0.7303–0.7374) |
| AUPRC | 0.8056 |
| Brier Score | 0.2115 |

### OOD Robustness
| Input | Mean Confidence | Correct % |
|-------|----------------|-----------|
| Normal EEG | 0.564 | 57.8% |
| Random Noise | 0.134 | N/A |
| Zero EEG | 0.139 | N/A |

### Selective Prediction (Conformer)
| Threshold τ | Coverage % | Accuracy % |
|------------|-----------|------------|
| 0.50 | ~65% | ~75% |
| 0.70 | 12.32% | 94.94% |

---

## 5. Falsification Controls (Phase 5)

| Control | Trial Acc | Window Acc |
|---------|-----------|------------|
| Standard (no tampering) | 71.88% | 57.69% |
| True Audio Permutation | 51.56% | 49.30% |
| Within-Subject Permutation | 48.44% | 49.06% |
| Cross-Subject Permutation | 49.69% | 50.29% |
| Gaussian Envelope | 54.69% | 50.63% |
| Zero EEG | 55.63% | 50.91% |
| Random EEG | 50.94% | 50.58% |
| Circular Shift 2s | 51.56% | 50.23% |
| Circular Shift 10s | 50.31% | 50.04% |
| Label Shuffle | 50.31% | 49.71% |

---

## 6. Cross-Dataset Transfer (Phase 10)

| Protocol | DTU Accuracy |
|----------|-------------|
| Accumulated Pearson (DTU baseline) | 68.24% |
| Majority Vote (KUL standard) | 54.26% |

### KUL S1 Zero-Shot (DTU-Trained MatchNet)

| Window Length | Window Acc | Trial Acc | Conf AUROC |
|--------------|-----------|-----------|------------|
| 30s | 75.8% | 100.0% | NaN |
| 20s | 71.4% | 95.0% | 0.952 |
| 15s | 69.1% | 95.0% | 0.814 |
| 10s | 66.8% | 90.0% | 0.812 |
| 5s | 60.1% | 90.0% | 0.729 |
| 2s | 54.3% | 75.0% | 0.612 |

---

## 7. SHAP Feature Importance

| Feature | Importance |
|---------|-----------|
| margin | 0.42 |
| rolling_std_margin | 0.35 |
| sim_chosen | 0.12 |
| trial_consistency | 0.08 |
| sim_unchosen | 0.03 |

---

## 8. Product Metrics (Phase 17)

### Phase 17.2 — Verified Controller Metrics

| Metric | Value |
|--------|-------|
| True Switches | 2 |
| False Switches | 2 |
| Precision | 50.0% |
| Coverage (Correct Lock Time) | 92.58% |

### Phase 17.3 — Redesigned UX Metrics

| Metric | Value |
|--------|-------|
| Audible False Switches/hr | 22.53 |
| Decision Availability | 99.63% |
| Correct Lock Coverage | 84.48% |
| Acquisition Latency | 4.99s |
| Switch/Recovery Latency | 25.41s |

---

## 9. Architecture Parameter Counts

| Architecture | Total Params | EEG Encoder | Audio Encoder | Head |
|-------------|-------------|-------------|---------------|------|
| ContrastiveMatchNet | 50,928 | 2,320 (4.6%) | 48,608 (95.4%) | — |
| TemporalCNN | ~69,000 | — | — | — |
| AAD-Conformer | ~2,083,000 | — | — | — |
| EEGNet (standalone) | 2,320 | — | — | 17 |
| AudioEncoder | 48,608 | — | — | — |
| XGBoost (Confidence) | 100 trees, depth 3 | — | — | — |

---

## 10. Statistical Tests (Run 1 Publication)

| Test | Statistic | p-value |
|------|-----------|---------|
| Paired t-test (Conformer vs Ridge) | t = 7.65 | p = 7.65×10⁻⁶ |
| Wilcoxon signed-rank | W = ? | p = 6.10×10⁻⁵ |
| Cohen's d | 1.6642 | — |
| Coefficient of Variation (5 seeds) | 4.0% | — |
