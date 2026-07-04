# MASTER FAILURES — Every Failed Hypothesis, Bug, and Dead End

> Failures are the most valuable artifacts in this repository. Each one directly informed the next design decision.

---

## Category 1: Data Leakage Bugs

### F-01: Validation Split Contamination (MatchNet v1)
- **Symptom**: 95%+ validation accuracy
- **Root Cause**: Validation data split at window level across all subjects. With 50% overlap between consecutive 3s windows, windows from the same trial appeared in both train and validation sets.
- **Duration**: ~3 weeks of wasted Kaggle GPU compute
- **Fix**: Strict LOSO — entire subjects held out
- **Impact**: Accuracy dropped from 95% to ~69% (genuine)
- **Lesson**: NEVER split windows randomly. Split at the subject level.

### F-02: The Negative Sampling Trap (MatchNet v2)
### F-02: Scipy IO Array Parsing Illusion
- **Failure**: The entire project was stalled for days because the target labels for AASD appeared to be missing or corrupted (with `latency == type`). The model subsequently trained on fabricated "Constant Left" labels, leading to test collapse.
- **Root Cause**: The helper function `get_ev_attr` used `hasattr(ev, 'latency')`. When `scipy.io.loadmat` parsed the EEGLAB struct as an array of arrays (`numpy.ndarray`) instead of an object, `hasattr` failed, and the function silently fell back to returning `ev[0]` (the event code) for all attribute queries.
- **Resolution**: Bypassed the helper function entirely. Mapped array indices natively (`[0]=type`, `[1]=latency`, `[4]=epoch`), which perfectly restored the down-to-the-millisecond ground-truth timing.
- **Source**: Phase 28 Forensic Ground-Truth Verifier

### F-03: Temporal Overfitting Without Margin Lossphic failure.

### F-04: Phase 17.2 NaN Equality Bug
- **Symptom**: Inflated False Switch Rate
- **Root Cause**: `np.nan != np.nan` evaluates `True` in pandas, causing NaN→NaN transitions (continuous uncertainty) to be counted as false switches
- **Fix**: Explicit NaN-aware equality checking
- **Lesson**: Always use `pd.isna()` for NaN comparisons. Never use `!=`.

---

## Category 2: Architecture Failures

### F-06: Temporal CNN LOSO Collapse
- **Architecture**: TemporalCNNAAD (~69,000 params) — Conv1d stem → multi-resolution parallel Conv1d → residual temporal blocks
- **Within-Subject**: ~70%+ (proving it can learn)
- **LOSO**: 50–55% (worse than linear Ridge baseline)
- **Shuffled Labels**: 45.83% (below chance, confirming severe memorization)
- **Root Cause**: The reconstruction objective (mapping EEG → envelope) is ill-posed. TCN memorized subject-specific noise profiles.
- **Impact**: Proved that architecture capacity is NOT the problem; the training objective is.
- **Source**: [temporal_cnn_loso_summary.json](file:///C:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/analysis/summaries/temporal_cnn_loso_summary.json)

### F-06: VLAAI-Lite LOSO Failure
- **Result**: ~50–55% LOSO
- **Same root cause as F-05**: Reconstruction objective + subject memorization
- **Status**: ABANDONED

### F-07: EEGNet-TCN LOSO Failure
- **Result**: ~50–55% LOSO
- **Same root cause as F-05**: Three different deep architectures all hit the same wall
- **Impact**: This triple failure (TCN, VLAAI-Lite, EEGNet-TCN) proved the problem was the OBJECTIVE, not the ARCHITECTURE, motivating the shift to contrastive learning
- **Status**: ABANDONED

### F-08: Ridge 8ch Summary Files (Empty)
- **Observation**: `ridge_loso_8ch_summary.json` and `ridge_loso_2ch_summary.json` both show trial_accuracy=0.0 and empty per_subject arrays
- **Root Cause**: Appears the alternative-channel ridge evaluations either crashed or were not completed
- **Status**: Only the 2-channel ridge with lags=48 produced valid results (55.19%)

---

## Category 3: Confidence & Estimation Failures

### F-09: Raw EEG Artifact Detection CNN
- **Hypothesis**: Build a secondary CNN to examine raw 8-channel EEG and predict whether it contains EMG artifacts
- **Result**: Suffered massive spatial leakage — learned to recognize subject-specific background noise baselines rather than generic artifact shapes
- **Root Cause**: Raw EEG features are too subject-specific for cross-subject confidence
- **Lesson**: Confidence must be derived from the LATENT space, not the raw input

### F-10: Bayesian Neural Networks (MC Dropout)
- **Hypothesis**: Run MatchNet multiple times with random dropout to estimate epistemic uncertainty
- **Result**: Discarded without implementation
- **Root Cause**: Computationally infeasible. A single forward pass is already near the budget of a hearing aid DSP. 30+ stochastic passes × 3s windows = physically impossible on battery-powered edge hardware.
- **Lesson**: Theoretical elegance means nothing if it can't run on the target device

### F-11: Softmax-Based Confidence
- **Hypothesis**: Use maximum softmax probability as confidence
- **Result**: Not applicable
- **Root Cause**: ContrastiveMatchNet produces continuous similarity scores in a geometric space. There is no classification head and no softmax distribution.
- **Lesson**: Architecture-aware confidence design is mandatory

### F-12: Confidence Head Dynamic Range Collapse
- **Symptom**: Confidence outputs compressed to [0.35, 0.52] — essentially uninformative
- **Root Cause**: 
  1. BCE loss penalizes extreme predictions; optimal strategy to minimize BCE on noisy data is to output the mean empirical accuracy
  2. 46.8% of ReLU neurons are dead (capacity collapse)
  3. Shortcut learning: 64-D z_pool ignored in favor of 1-D margin scalars (faster gradient path)
- **Impact**: Confidence head was functioning as a sigmoid scaling function on margin, not a true uncertainty estimator
- **Source**: [phase13_confidence_architecture_review.md](file:///C:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/docs/reviews/phase13_confidence_architecture_review.md)

### F-13: Confidence Leakage (0.99 AUROC → 0.59)
- **Symptom**: Initial confidence AUROC appeared to be ~0.99
- **Root Cause**: Confidence evaluation was contaminated — evaluation used data that was also used to calibrate thresholds
- **After correction**: AUROC for high-confidence failure prediction = 0.59 (the information limit)
- **Lesson**: Confidence estimation requires its own strict leakage prevention protocol

---

## Category 4: Evaluation & Protocol Failures

### F-14: Majority Vote on ~53% Window Accuracy
- **Symptom**: Phase 11 selective AAD yielded 0.0% trial accuracy on KUL S1
- **Root Cause**: KUL S1 window accuracy was ~53%. With near-chance windows, majority voting across a trial is pure coin-flip. The threshold logic was misaligned with majority vote mechanics.
- **Impact**: Proved that naive window voting CANNOT work. Motivated Phase 12 (temporal evidence accumulation).
- **Source**: [phase11_project_state.md](file:///C:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/docs/reports/phase11_project_state.md)

### F-15: Aggregation Method Confusion (68.24% vs 54.26%)
- **Symptom**: Phase 10 cross-dataset evaluation produced two contradictory accuracy numbers
- **Root Cause**: NOT a model error. Two mathematically valid but different aggregation protocols: Accumulated Pearson (DTU convention) = 68.24%, Majority Vote (KUL convention) = 54.26%.
- **Impact**: Could have been misinterpreted as a model failure. Required complete repository forensics to trace the exact code paths.
- **Source**: [phase10_cross_dataset_evaluation.md](file:///C:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/docs/reports/phase10_cross_dataset_evaluation.md)

### F-16: Phase 17.2 Undercounting Switches
- **Symptom**: Phase 17.2 reported only 2 false switches, but Phase 17.3 found 22.53 audible false switches per hour
- **Root Cause**: Phase 17.2 counted internal controller state transitions. Many rapid oscillations at the 46.875ms tick level were invisible because NaN→NaN transitions were excluded. Phase 17.3's Output State Machine (collapsing to 1-second stable states) revealed the true user-perceived behavior.
- **Lesson**: Internal controller metrics ≠ user-perceived metrics. Always measure from the output.

---

## Category 5: Preprocessing & Compatibility Failures

### F-17: KUL Single-Band Envelope Failure
- **Symptom**: KUL zero-shot transfer yielded ~50% accuracy (chance)
- **Root Cause**: Audio was preprocessed as a single broadband envelope (192,) instead of 28-band Gammatone (28, 192). MatchNet's AudioEncoder hardcodes `in_channels=28`.
- **Fix**: Rebuilt 28-band Gammatone filterbank (ERB-spaced, 50–8000 Hz) for KUL audio
- **After fix**: Accuracy jumped to 75.8% (30s windows)
- **Source**: [REPOSITORY_MODEL_LINEAGE.md](file:///C:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/docs/REPOSITORY_MODEL_LINEAGE.md)

---

## Failure Narrative (Chronological Flow)

```
Ridge Baseline (65–69%)
    ↓ "Can deep learning do better?"
TCN Reconstruction (50–55%) — FAILURE F-05
    ↓ "Reconstruction is ill-posed"
Contrastive MatchNet v1 (95%) — LEAKAGE F-01
    ↓ "Fix validation split"
Contrastive MatchNet v2 (95%) — LEAKAGE F-02
    ↓ "Fix negative sampling"
Contrastive MatchNet v3 (69%) — GENUINE
    ↓ "Not clinically viable. Can we know WHEN it fails?"
Raw EEG Confidence CNN — FAILURE F-09
    ↓ "Raw features are subject-specific"
MC Dropout — FAILURE F-10
    ↓ "Computationally infeasible"
Geometric Confidence (AUROC 0.80) — SUCCESS
    ↓ "Is this real?"
Confidence appeared 0.99 AUROC — LEAKAGE F-13
    ↓ "Information limit discovered at 0.59"
Learned Confidence Head — FAILURE F-12
    ↓ "Dynamic range collapsed"
Evidential DL Proposed — PENDING
```

> **Total Failures Cataloged**: 17
> **Each one directly informed the next design decision.**
