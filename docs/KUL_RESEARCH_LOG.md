# KUL Research Log

This document serves as the central experiment log for all KUL dataset experiments.

## Experiment: KUL Ridge LOSO Baseline

### Dataset
* **Dataset:** KUL
* **Subjects:** 16
* **Evaluation:** LOSO
* **Model:** Classical Ridge Regression (EEG-only)
* **Preprocessing:** Cached preprocessed KUL dataset

### Legacy Trial Accuracy

| Window | Accuracy |
| ------ | -------: |
| 2 s    |    54.7% |
| 5 s    |    56.6% |
| 10 s   |    55.6% |
| 15 s   |    55.0% |
| 30 s   |    55.6% |

**Observations:**
* Best performance occurs at 5 seconds.
* Overall Ridge baseline is approximately 55–57%.
* Window length has relatively little influence after 5 seconds.

**Conclusion:**
This becomes the official classical baseline for KUL.

---

## Experiment: Ridge Sanity Checks

**Subjects Evaluated:** S1, S5, S11, S13

### Results

| Test             |  Mean |
| ---------------- | ----: |
| Baseline         | 55.0% |
| Zero EEG         | 45.0% |
| Mismatched EEG   | 47.5% |
| Shuffle EEG      | 46.2% |
| Mismatched Audio | 45.0% |
| Shift 5 s        | 37.5% |
| Shift 10 s       | 41.2% |
| Shift 20 s       | 42.5% |
| Swap Labels      | 35.0% |
| Random Train     | 41.2% |

**Observations:**
* Removing EEG causes performance to collapse toward chance.
* Mismatching EEG also collapses performance.
* Destroying temporal alignment between EEG and audio causes large degradation.
* Random training pairings fail to learn meaningful decoding.
* The Ridge decoder appears to rely on genuine EEG–audio relationships instead of exploiting shortcuts.

**Conclusion:**
The Ridge baseline passes the sanity suite and is considered a trustworthy baseline for future comparisons.

---

## Experiment: Architecture 1 - Temporal CNN (TCNN)

### Architecture Configuration
* **Type:** EEG-only Binary Classifier (Left vs Right ear spatial attention)
* **Backbone:** 1D Temporal CNN (3 conv blocks, Global Average Pooling)
* **Training:** Class-balanced (56 Class 1 vs 56 Class 2)
* **Evaluation:** LOSO with Early Stopping on held-out validation set

### LOSO Results (10s Window)

| Fold (Held-Out) | Window Acc | Trial Acc |
| --------------- | ---------: | --------: |
| S1              |     63.27% |    80.00% |
| S10             |     62.17% |    80.00% |
| S11             |     63.78% |    80.00% |

**Observations:**
* The trial accuracy is stuck at exactly 80.00% for every single fold.
* The training logs reveal a massive underlying dataset imbalance in KUL: 80% of all trials are Class 1 (Left Ear), and only 20% are Class 2 (Right Ear) [224 vs 56 trials].
* The validation trial accuracy violently oscillates between 20.0% and 80.0% between epochs, meaning the model is collapsing and predicting a single constant class for the entire validation set. 
* Because the validation set is unbalanced (80% Class 1), Early Stopping strictly favors epochs where the model collapses to constantly predicting Class 1 (since it lowers cross-entropy loss).
* As mathematically predicted in the `kul_80_percent_anomaly.md` investigation, a collapsed model evaluating an 80/20 imbalanced dataset will achieve exactly 80.0% accuracy on every subject with 0% standard deviation.

**Conclusion:**
The TCNN baseline fails. The EEG-only binary spatial classification task is completely ruined by the 80/20 class imbalance in the KUL dataset, causing model collapse. We cannot use spatial binary classification on this dataset without a different approach.

---

## Experiment: TCNN with Window-Level Balancing (Forensic Audit)

**Date:** June 28, 2026

### What was changed
A massive data audit revealed that although the training *trials* were balanced (56 Track 1 vs 56 Track 2), the generated *windows* were severely imbalanced (33% Track 1 vs 67% Track 2) because the KUL dataset stories for Track 2 are systematically twice as long as Track 1. 

The training pipeline was modified to generate all chunks first, and then explicitly downsample the majority class to guarantee a perfectly balanced 50/50 dataset for the DataLoader.

### Before/After Window Distributions
* **Before:** Track1 windows: 5241, Track2 windows: 10724 (33% / 67%)
* **After:** Track1 windows: 5241, Track2 windows: 5241 (50% / 50%)

### Final LOSO Accuracy & Forensic Checks
Despite perfectly balanced training windows, the model *still* collapses:

| Fold (Held-Out) | Window Acc | Trial Acc | Behavior |
| --------------- | ---------: | --------: | :------- |
| S1              | ~64%       | 75.0%     | Predicted Track 1 for 19/20 trials. |
| S10             | ~64%       | 80.0%     | Predicted Track 1 for 20/20 trials. |
| S13             | ~64%       | 20.0%     | Predicted Track 2 for 20/20 trials. |

**Observations:**
* Window balancing successfully ensures the model sees exactly 50/50 classes during training.
* Despite this, the TCNN still collapses into a near-constant function (predicting `11111...` or `22222...` for all windows).
* The 80% trial accuracy is therefore **NOT** caused by window imbalance in the training set. 
* The root cause is that the TCNN architecture (EEG-only) is fundamentally incapable of learning generalized spatial attention features across subjects in this configuration. Because it learns nothing (SNR ≈ 0), its output is purely driven by initialization noise and optimization drift, and the 80/20 unbalanced validation set forces Early Stopping to capture the model precisely when it drifts into a collapsed state.

**Conclusion:**
The TCNN architecture cannot solve the EEG-only binary spatial attention task on KUL. The collapse is structural, not a sampling artifact. We must investigate the evaluation logic, feature distribution shift, or move to a different architectural paradigm.
