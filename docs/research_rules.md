# Research Validation Rules

This document details the scientific and methodological validation rules enforced by the EEG Research Validator in `server.py`.

---

## 🧪 Enforced Rules

### 1. Leave-One-Subject-Out (LOSO) Split Leakage
- **Violation**: Splitting dataset randomly across samples instead of subjects.
- **Why**: EEG signals display high inter-subject variability. Splitting trials randomly across train and test sets leads to massive subject leakage (the model memorizes subject characteristics rather than learning generalized features).
- **Trigger**: Checked if file contains both the keywords `"loso"` and `"train_test_split"`.

---

### 2. Temporal Dimension Collapse before InfoNCE
- **Violation**: Calling `mean(dim=1)` or collapsing temporal dimensions before computing similarity.
- **Why**: InfoNCE loss in multi-modal models (e.g., EEG-Audio contrastive learning) assumes temporal alignment is preserved when matching sequence dimensions. Collapsing time steps prematurely removes local dynamic alignment.
- **Trigger**: Checked if file contains `"sim = einsum"` and `"mean(dim=1)"`.

---

### 3. Dataset Compatibility Warnings
- **Violation**: Referencing both DTU and KUL loaders inside a single script.
- **Why**: Preprocessing rates, channel counts, and trial structures differ between datasets, which can lead to loader bugs or mismatch errors.
- **Trigger**: Checked if file contains both `"dtu"` and `"kul"`.
