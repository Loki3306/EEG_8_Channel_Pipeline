# Confidence-Aware MatchNet AAD Roadmap

## Goal
Transform the current MatchNet pipeline:
```text
EEG + Audio → MatchNet → Speaker A / Speaker B
```
into a confidence-aware system capable of selective decoding:
```text
EEG + Audio → MatchNet → Prediction → Confidence → Risk Estimate → Optional Abstention
```
This will be achieved while maintaining strict LOSO evaluation.

---

## Phase 0 — Infrastructure Preparation
**Objective:** Create the confidence evaluation framework before changing any model architectures.

### Task 0.1: Baseline Freeze
- Freeze the current MatchNet baseline.
- Store: Current LOSO accuracy, Per-subject accuracy, Window-wise accuracy, Model checkpoint, Predictions, and Similarity scores.

### Task 0.2: Inference Modification
Modify the inference pipeline from a strict `argmax(sim_A, sim_B)` to save all decision components.
- **Required Data:** `sim_A`, `sim_B`, `prediction`, `label`, `subject_id`, `trial_id`, `window`.

**Deliverable:**
A comprehensive baseline CSV: `subject | trial | window | sim_A | sim_B | prediction | label | correct`
This dataset becomes the foundation for all subsequent confidence experiments.

---

## Phase 1 — Confidence Benchmarking
**Objective:** Determine whether the current MatchNet already contains useful confidence information without any retraining or architecture changes.

### Method 1: Similarity Margin
Compute: `margin = abs(sim_A - sim_B)`
- Sort windows by confidence margin.
- Bin margins (e.g., `0.0-0.1`, `0.1-0.2`) and compute accuracy per bin.
- **Hypothesis:** Higher margin correlates with higher accuracy.

### Method 2: Softmax Confidence
Compute: `p = softmax([sim_A, sim_B])`
- `confidence = max(p)`

### Deliverables:
1. **Plot 1:** Confidence vs. Accuracy.
2. **Plot 2:** Confidence Histogram (Correct vs. Incorrect distributions).
3. **Plot 3:** Confidence Distribution (Strong subjects vs. Weak subjects).

**Success Criteria:** Confidence strongly correlates with correctness. If not, stop the confidence project here.

---

## Phase 2 — Reliability Analysis
**Objective:** Rigorously determine whether confidence predicts correctness.

### Metrics:
1. **AUROC:** Can confidence separate Correct vs. Incorrect predictions? (Target: `AUROC > 0.7`)
2. **AUPRC:** Useful secondary metric for assessing precision-recall if errors become rare.
3. **Brier Score:** Measures the overall quality and calibration of the confidence scores.
4. **Expected Calibration Error (ECE):** Predicted Confidence vs. Observed Accuracy.
5. **Reliability Diagram:** Plot of Confidence vs. Actual Accuracy (perfect calibration = diagonal line).

**Deliverables:** Metric report containing AUROC, AUPRC, ECE, Brier, and Reliability Diagram.

**Decision Gate 1:** If confidence is meaningful, proceed. Otherwise, confidence research ends.

---

## Phase 3 — Selective AAD
**Objective:** Allow the model to refuse highly uncertain predictions (Abstention).

### Implementation:
Choose threshold `T`. If `confidence < T`, abstain. Else, predict.
- Sweep `T` from `0.1` to `0.9`.

### Metrics:
- **Coverage:** Accepted Samples / Total Samples
- **Risk:** 1 - Accuracy (on accepted samples)
- **Accuracy:** Accuracy on accepted samples only.

**Deliverables:**
1. Coverage vs. Accuracy report (e.g., 100% coverage → 68% acc; 40% coverage → 91% acc).
2. Risk-Coverage Curve.
3. AURC (Area Under Risk-Coverage Curve).

**Decision Gate 2:** If selective decoding yields major gains, this is immediately a publication candidate (Selective Auditory Attention Decoding).

---

## Phase 4 — Confidence Calibration
**Objective:** Convert abstract confidence scores (e.g., Margin = 0.7) into true probabilities (e.g., 82% chance prediction is correct).

### Methods:
1. Temperature Scaling
2. Platt Scaling
3. Isotonic Regression

**Deliverable:** Calibrated confidence scores mapped against ECE and Brier scores (Before vs. After).

---

## Phase 5 — MC Dropout Uncertainty
**Objective:** Estimate epistemic uncertainty dynamically.

### Implementation:
- Modify MatchNet to keep Dropout layers active during inference.
- Run 30 forward passes per sample.
- Store: `mean_prediction`, `variance`, and `entropy`.

**Deliverables:** Compare MC Dropout AUROC against standard Margin Confidence.
**Decision Gate 3:** Keep MC Dropout only if it significantly outperforms Margin.

---

## Phase 6 — Confidence Head
**Objective:** Teach the network explicitly to predict its own correctness.

### Architecture:
```text
EEG Encoder → Latent Space → [Prediction Head] & [Confidence Head]
```
- **Labels:** After prediction, `correct = 1`, `incorrect = 0`.
- **Loss:** `L_total = L_matchnet + λ * L_confidence`
- Confidence Head learns: `P(prediction is correct)`

**Deliverables:** Comparison of Margin vs. MC Dropout vs. Confidence Head.

---

## Phase 7 — Subject-Aware Confidence (Highest Priority)
**Objective:** Address the hypothesis that weak subjects (S7, S15, S10, S11) are Out-Of-Distribution (OOD) subjects.

### Step 7.1: Subject Embedding Extraction
Extract representations:
- Covariance matrices
- PSD features
- Latent MatchNet embeddings

### Step 7.2: Distance Metrics
Compute distances between individual subjects and the training population:
- Mahalanobis Distance
- Riemannian Distance (Covariance Geometry)
- PSD Distance (Theta/Alpha/Beta profiles)
- Domain Shift Metrics (CORAL, MMD, Wasserstein)

### Core Experiment:
Predict **LOSO Accuracy** using **Subject Distance**.

**Deliverables:** Scatter plots of Distance vs. Accuracy.
**Success Criterion:** Large distance correlates with poor LOSO performance. If proven true, this constitutes a major publication opportunity.

---

## Phase 8 — EEG Quality Confidence
**Objective:** Estimate signal reliability before MatchNet processing.

### Features:
- PSD Stability, Signal Variance, Entropy, Covariance Stability, Artifact Score, Spectral Flatness, Channel Correlation.

**Deliverable:** Pre-decoding EEG Quality Score capable of predicting MatchNet failure.

---

## Phase 9 — Confidence Fusion
**Objective:** Combine all reliability metrics into a final unified system.

### Fusion Architecture:
- **Inputs:** Margin Confidence, Calibrated Confidence, MC Dropout Variance, Confidence Head Output, Subject Distance, EEG Quality Score.
- **Network:** Feature Vector → MLP → Reliability Score.

**Final Evaluation:**
Compare AUROC, ECE, and AURC across:
- Margin, Softmax, MC Dropout, Confidence Head, Subject-Aware, and Fusion.

---

## Execution Protocol & 4-Week Plan

**Execution Order:**
Do NOT build everything at once. Adhere to strict decision gates.
1. `Phase 0 → Phase 1 → Phase 2 → Phase 3`
   - **STOP & Evaluate.**
2. If successful: `Phase 4 → Phase 6 → Phase 7`
   - **STOP & Evaluate.**
3. Only if proven valuable: `Phase 5 → Phase 8 → Phase 9`

### Weekly Schedule

#### Week 1
- Freeze MatchNet baseline.
- Export predictions/similarities.
- Compute Margin and Softmax confidences.

#### Week 2
- Compute AUROC, ECE, Brier.
- Generate Reliability diagrams.

#### Week 3
- Implement Selective AAD.
- Generate Risk-Coverage curves and AURC.

#### Week 4
- Confidence calibration.
- Prototype Confidence head.

*At the end of Week 4, we will determine if confidence is a viable signal. If yes, Phase 7 (Subject-Aware Confidence) becomes the primary research direction due to its high novelty-to-effort ratio.*
