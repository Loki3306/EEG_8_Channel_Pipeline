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
