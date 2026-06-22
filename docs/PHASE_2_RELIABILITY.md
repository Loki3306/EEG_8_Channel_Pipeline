# Phase 2: Reliability Analysis

## Phase Status

- [x] Step 1.1 Export Predictions
- [x] Step 1.2 Margin Confidence
- [x] Step 2.1 AUROC / AUPRC
- [x] Step 2.2 Selective AAD Pilot
- [ ] Step 2.3 Calibration Analysis

---

# Experiment Log

## Step 1.1 — Export Validation
- **Result**: 5400 evaluation windows, 69.02% MatchNet LOSO accuracy
- **Status**: PASS

## Step 1.2 — Margin Confidence
- **Mean Margin**: Correct (0.0749) vs Incorrect (0.0478)
- **Margin Binning**: Higher margin consistently corresponds to higher accuracy (monotonic growth from 57.60% at 0.00-0.05 to 100.00% at 0.25-0.30).
- **Conclusion**: Margin contains meaningful confidence information.
- **Status**: PASS

## Step 2.1 — Reliability Metrics
- **AUROC**: 0.6601
- **AUPRC**: 0.8109
- **Baseline Precision**: 0.6902
- **Conclusion**: Margin confidence is significantly better than chance (useful, but not a strong confidence signal).
- **Status**: PASS

## Step 2.2 — Selective AAD
- **Findings**: Rejecting low-confidence predictions improves accuracy (100% Coverage -> 69.02%; 30% Coverage -> 83.83%).
- **Conclusion**: Margin confidence helps, but does not fully explain prediction failures (improvement is moderate rather than dramatic).
- **Status**: PASS

---

## Scientific Conclusions So Far
1. Confidence information exists inside MatchNet.
2. Similarity margin can act as a confidence score.
3. Margin confidence is useful but limited.
4. The results support investigating stronger confidence estimators.
5. Subject-aware confidence remains strongly justified.
