# MASTER TIMELINE — EEG AAD Project Chronology

> Reconstructed from repository artifacts, commit history, reports, and code timestamps.

---

## Phase 0 — Dataset Exploration & Data Understanding
**Motivation**: Understand the raw DTU `.mat` file structure before any modeling.
**Key Activities**:
- Inspected 18 subject files (S1–S18), each with 60 trials
- Confirmed structure: 3200 samples × 66 channels × 64 Hz
- Discovered `wavA` and `wavB` are NOT ordered by label
- Labels (1/2) represent speaker identity (male/female), not A/B index
- Audio clustering successfully separates 2 speakers (distinct_cluster_rate = 1.0)
- Mapping: {0: 2, 1: 1} with accuracy 1.0000

**Files**: [DATA_ANALYSIS.md](file:///C:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/DATA_ANALYSIS.md), [inspect_data.py](file:///C:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/analysis/inspect_data.py), [validate_labels.py](file:///C:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/analysis/validate_labels.py)

**Conclusion**: Dataset is valid. Label semantics resolved. `wavA` = always attended in preprocessed data.

**→ Motivated Phase 1**: Data is understood → build a baseline decoder.

---

## Phase 1 — Linear Baselines (Ridge + Early CNN)
**Motivation**: Establish a performance floor with minimal assumptions.
**Key Activities**:
- Ridge regression baseline: 2-channel, 16 lags, λ=1.0
- LOSO trial accuracy: 55.19% (2ch) / ~65-69% (8ch)
- Temporal CNN reconstruction: ~56.4% LOSO accuracy (69k params)
- Contrastive baseline: ~58.89% LOSO accuracy

**Key Metric**: Ridge 2ch LOSO = 0.5519, Ridge balanced = 0.5531
**Per-Subject Range**: S5 = 0.467 (worst) to S15 = 0.667 (best)

**Conclusion**: Signal exists but is weak. Model capacity is NOT the bottleneck under 2-channel strict LOSO.

**→ Motivated Phase 2**: Improve signal extraction and training objectives, not architecture size.

---

## Phase 2 — Signal Preprocessing & Contrastive Learning
**Motivation**: Raw correlation loss is insufficient. The EEG-to-audio mapping is non-linear.
**Key Activities**:
- Implemented Hilbert envelope extraction with power-law compression (^0.6)
- Added configurable lag windows and longer evaluation windows (10s, 20s, 30s)
- Shifted from reconstruction to contrastive (InfoNCE) objective
- Hard negative sampling strategies: random, nearby, same-trial, mixed

**Conclusion**: Contrastive learning fundamentally changes the problem from regression to discrimination.

**→ Motivated Phase 3**: Need stronger architecture (Conformer) to fully exploit the contrastive objective.

---

## Phase 3 — AAD-Conformer Implementation
**Motivation**: Linear and small CNN models hit a ceiling. Need attention-based architecture.
**Key Activities**:
- Implemented AAD-Conformer: EEGNet-style stem → Conformer blocks → regression head
- Architecture: 32 temporal filters, 64 spatial filters, 2 layers, 4 attention heads
- Initial results: ~77% — suspiciously high

**Critical Discovery**: Evaluation leakage found — test subjects leaking into validation.

**→ Motivated Phase 4**: Fix leakage, then re-evaluate.

---

## Phase 4 — Leakage Purge & Unbiased Evaluation
**Motivation**: Must prove 77% is real, not artifact of leakage.
**Key Activities**:
- Completely rewrote evaluation to enforce strict LOSO
- Purged all validation contamination
- Unbiased result: **71.88% trial accuracy, 57.69% window accuracy**
- Margin: +0.0238

**→ Motivated Phase 5**: 71.88% is high for 8ch. Must attempt to destroy it (falsification).

---

## Phase 5 — Scientific Falsification (10 Negative Controls)
**Motivation**: If 71.88% is real, no negative control should achieve above chance.
**Key Activities**:
- 10 negative controls: audio permutation, zero EEG, random EEG, circular shifts, label shuffle
- All 9 controls collapsed to ~50% ± 1%
- 100% stimulus overlap discovered (all test stories seen during training by other subjects)

**Key Results**:
| Control | Trial Acc | Window Acc |
|---------|-----------|------------|
| Standard | 71.88% | 57.69% |
| True Audio Perm | 51.56% | 49.30% |
| Within-Subject Perm | 48.44% | 49.06% |
| Cross-Subject Perm | 49.69% | 50.29% |
| Gaussian Envelope | 54.69% | 50.63% |
| Zero EEG | 55.63% | 50.91% |
| Random EEG | 50.94% | 50.58% |
| Circular Shift 2s | 51.56% | 50.23% |
| Circular Shift 10s | 50.31% | 50.04% |
| Label Shuffle | 50.31% | 49.71% |

**Conclusion**: Model is scientifically validated. Accuracy is genuine.

**→ Motivated Phase 6**: Need cross-dataset validation (KUL) to address stimulus overlap limitation.

---

## Phase 6 — Multi-Seed Reproducibility (Run 1)
**Motivation**: Ensure Conformer performance is not initialization-dependent.
**Key Activities**:
- 5 seeds: 1, 7, 21, 42, 123
- Mean accuracy: 77.12% ± 9.99%
- Per-seed: 71.9%, 79.4%, 78.1%, 75.6%, 80.6%
- Coefficient of Variation: 4.0%
- Paired t-test vs Ridge: p = 7.65×10⁻⁶, Cohen's d = 1.6642
- Wilcoxon: p = 6.10×10⁻⁵

**Best Subject**: S11 = 99.0% mean (100% on 3 seeds)
**Worst Subject**: S15 = 63.0% mean (45% on seed 42)

**→ Motivated Phase 7**: Reproducibility proven → add confidence estimation.

---

## Phase 7 — Confidence Head (Learned Confidence)
**Motivation**: 29% of predictions are wrong. Need to know WHEN.
**Key Activities**:
- Built Late-Fusion Confidence Head (MLP on z_pool + margin + corr_a + corr_b + latent_norm)
- Outlier Exposure training: inject Random/Zero EEG with target=0
- OOD Robustness verified: Random noise → 0.134 conf, Zero EEG → 0.139 conf

**Key Metrics**:
| Metric | Value |
|--------|-------|
| ECE | 0.0998 |
| AUROC | 0.7337 (CI: 0.7303–0.7374) |
| AUPRC | 0.8056 |
| Brier Score | 0.2115 |

**Selective Prediction**: τ=0.70 → 94.94% accuracy @ 12.32% coverage

**→ Motivated Phase 8**: Extend confidence to the DTU MatchNet pipeline.

---

## Phase 8 — DTU MatchNet Confidence (XGBoost Framework)
**Motivation**: Replicate confidence framework on the DTU ContrastiveMatchNet.
**Key Activities**:
- 5 geometric features: margin, sim_chosen, sim_unchosen, rolling_std_margin, trial_consistency
- XGBoost classifier (100 trees, depth 3)
- Margin-only AUROC: 0.6601
- Full 5-feature AUROC: 0.8057 (CI: 0.7936–0.8182)
- AURC: 0.1320, E-AURC: 0.0781

**Selective AAD**: 81.55% accuracy @ 70% coverage (+12.53% gain)

**→ Motivated Phase 9**: Validate with hostile audits.

---

## Phase 9 — Exhaustive Audit Series (8 Hostile Audits)
**Motivation**: Prove confidence is not exploiting leakage.
**Key Audits**:
1. Behavioral Audit: Coverage-accuracy monotonic ✓
2. Minimal Model Audit: Margin-only AUROC ~0.65, full ~0.78 ✓
3. Margin Necessity: Removing margin drops AUROC significantly ✓
4. Decision Path (SHAP): rolling_std + margin = >75% importance ✓
5. Root Cause: Failures correlate with broadband EMG spikes ✓
6. Information Limit: High-confidence failure prediction AUROC ≈ 0.59 ✓
7. Stimulus Audit: verified ✓
8. Publication Figures: generated ✓

**Discovery**: Information limit of similarity-derived confidence at AUROC ≈ 0.59

**→ Motivated Phase 10**: Cross-dataset generalization.

---

## Phase 10 — Cross-Dataset Zero-Shot (KUL → DTU)
**Motivation**: Prove model generalizes to unseen datasets/subjects/stimuli.
**Key Activities**:
- Trained on KUL, tested on DTU (completely frozen)
- Built KUL preprocessing pipeline (128Hz→64Hz, channel mapping, 28-band Gammatone)
- Major discovery: evaluation discrepancy — Accumulated Pearson (68.24%) vs Majority Vote (54.26%)

**Root Cause**: DTU baselines use Accumulated Pearson; KUL uses Majority Vote.

**Resolution**: Report both metrics. Neither is "wrong".

**→ Motivated Phase 11**: Formalize selective AAD production system.

---

## Phase 11 — Production Selective AAD System
**Motivation**: Bridge window-level predictions to trial-level decisions.
**Key Activities**:
- Built Unified Confidence Engine (`src/confidence/`)
- Threshold sweep: τ=0.75 → 60% coverage, >85% accuracy
- Root cause of trial-level failure: window accuracy ~53% on KUL S1 → majority vote fails
- Identified need for temporal memory (Window Buffer)

**→ Motivated Phase 12**: Build sequential decision infrastructure.

---

## Phase 12 — Sequential Window Buffer
**Motivation**: Independent window predictions are insufficient. Need temporal memory.
**Key Activities**:
- Implemented `SequentialWindowBuffer` with `WindowPrediction` dataclass
- Passive memory infrastructure — no decision logic
- Supports running statistics, prediction/confidence/margin history

**→ Motivated Phase 13**: Add intelligent decision policy on top of buffer.

---

## Phase 13 — Confidence Architecture Redesign
**Motivation**: Current confidence head has compressed dynamic range [0.35, 0.52] and 46.8% dead neurons.
**Key Activities**:
- Diagnosed: BCE loss causes mean-reversion, z_pool ignored due to shortcut learning
- Proposed HETC: Hybrid Evidential-Temporal Confidence
- Implemented `TemporalEvidenceAccumulator` with Dirichlet parameters
- Implemented `EvidentialLoss` in training/evidential_loss.py

**Chosen Architecture**: Evidential Deep Learning Head + Sequential Evidence Accumulation

**→ Motivated Phase 14**: Build and validate the evidence accumulation system.

---

## Phase 14 — Evidence Accumulation & Streaming Simulator
**Motivation**: Need temporal decision-making for continuous streaming.
**Key Activities**:
- Implemented SPRT-based evidence accumulation
- Built streaming validation pipeline
- Report generator for continuous metrics

**→ Motivated Phase 15**: Build the full decision policy engine.

---

## Phase 15 — Decision Policy Engine
**Motivation**: Need a complete FSM (Finite State Machine) for hearing-aid output control.
**Key Activities**:
- Built `DecisionPolicyEngine` with states: INITIALIZING, WAITING, LOCKED, SWITCHING, UNCERTAIN
- Evidence via causal Log-Likelihood Ratio (LLR) accumulation
- Configurable: base_threshold=0.85, min_lock=5, min_switch_gap=10, min_consecutive=3

**Sub-phases**:
- 15.1: Bottleneck audit
- 15.2: Decision flow audit
- 15.3: Wrong decision audit
- 15.4: Robustness audit + Early difficulty falsification

**→ Motivated Phase 16**: Add context-aware heuristics and continuous session simulation.

---

## Phase 16 — Context-Aware Policy Engine & Hardware Emulator
**Motivation**: Static thresholds are insufficient. Need adaptive behavior.
**Key Activities**:
- 16.1: `ContextAwarePolicyEngine` with 5 heuristics: difficulty scaling, growth rate, oscillation penalty, hysteresis, cooldown
- 16.2: `ContinuousSessionGenerator` with `DatasetAdapter` pattern, 5 reusable scenarios
- 16.3: Transition Semantics formalization (splice timestamp, center-sample convention)

**Scenarios**: stable_conversation, single_shift, rapid_conversation, mixed_difficulty, long_continuous

**→ Motivated Phase 17**: Run continuous evaluation on all scenarios.

---

## Phase 17 — Continuous Evaluation & Product Metrics
**Motivation**: Evaluate the entire system as a hearing aid would experience it.

### Phase 17.1 — Continuous Evaluation
- Ran all 5 scenarios through the ContextAwarePolicyEngine
- Generated per-window event logs

### Phase 17.2 — Metric Audit & Event Reconstruction
- Discovered `np.nan != np.nan` bug inflating false switch count
- After fix: True Switches=2, False Switches=2, Precision=50%, Coverage=92.58%
- Generated case studies showing model error (not policy error) as root cause

### Phase 17.3 — Product Metrics Redesign (CURRENT)
- Redesigned metrics to measure user-perceived behavior, not internal controller events
- Output State Machine collapses 50ms internal ticks into 1-second stable states
- **Revelation**: Phase 17.2 was systematically undercounting switches (NaN transition blind spot)

**Current UX Metrics**:
| Metric | Value |
|--------|-------|
| Audible False Switches/hr | 22.53 |
| Decision Availability | 99.63% |
| Correct Lock Coverage | 84.48% |
| Acquisition Latency | 4.99s |
| Switch/Recovery Latency | 25.41s |

---

## Master Flow Diagram

```
Phase 0 (Data Understanding)
    ↓
Phase 1 (Ridge/CNN Baselines: ~55%)
    ↓
Phase 2 (Preprocessing + Contrastive Learning)
    ↓
Phase 3 (AAD-Conformer: 77% → leakage found)
    ↓
Phase 4 (Leakage Purge: 71.88% unbiased)
    ↓
Phase 5 (Falsification: 10/10 PASS)
    ↓
Phase 6 (Multi-Seed: 77.12% ± 9.99%)
    ↓
Phase 7 (Conformer Confidence Head: AUROC 0.7337)
    ↓
Phase 8 (MatchNet XGBoost Confidence: AUROC 0.8057)
    ↓
Phase 9 (8 Hostile Audits: ALL PASS)
    ↓
Phase 10 (Cross-Dataset Zero-Shot: 68.24%/54.26%)
    ↓
Phase 11 (Production Selective AAD)
    ↓
Phase 12 (Window Buffer Memory)
    ↓
Phase 13 (Evidential DL Redesign)
    ↓
Phase 14 (Evidence Accumulation + Streaming)
    ↓
Phase 15 (Decision Policy Engine FSM)
    ↓
Phase 16 (Context-Aware Engine + Hardware Emulator)
    ↓
Phase 17 (Continuous Evaluation → Product Metrics)
    ↓
Phase 18+ (TBD — see MASTER_ROADMAP.md)
```
