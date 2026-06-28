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
