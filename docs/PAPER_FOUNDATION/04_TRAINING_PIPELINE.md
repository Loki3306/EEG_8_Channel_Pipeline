# 04 Training Pipeline

## Cross-Validation Strategy
The model was trained and evaluated using a **Leave-One-Subject-Out (LOSO)** cross-validation protocol on the DTU dataset.
- For 18 subjects, 18 separate models were trained.
- Each fold holds out all trials for 1 subject as the test set.
- A subset of the remaining 17 subjects is used for validation (early stopping), and the rest for training.

## Checkpoint Selection and Early Stopping
- **Metric**: Validation Loss (InfoNCE).
- **Early Stopping**: Training was halted if the validation loss did not improve for a set number of epochs (patience).
- The best checkpoint based on validation loss was saved as `matchnet_fold_S<X>_best.pth` and used for the final test set evaluation.

## Bug Fixes & Discovered Issues
### The Negative Sampling Leakage Bug
- **Issue**: During early iterations of MatchNet development, a severe data leakage bug was found. The data loader was creating negative samples (unattended audio) by randomly pairing EEG chunks with audio chunks from *different trials* or *different subjects*.
- **Impact**: The model learned to perform trial-identification or subject-identification rather than true auditory attention decoding.
- **Fix**: Negative sampling was constrained rigidly. The unattended audio `z_b` is strictly defined as the concurrent, unselected audio stream from the *exact same trial* as the EEG.

### Checkpoint Mismatches
- **Issue**: In Phase 6 (KUL transfer), there was an initial risk of using the wrong checkpoint. The repository contained models like `eegnet_tcn`, `vlaai`, and `atcnet`.
- **Fix**: A full repository archaeology was performed to map the production `subject_distance_predictions.csv` directly to the `models/matchnet.py` `ContrastiveMatchNet` definition.

### Global Normalization Mismatch
- **Issue**: During cross-dataset transfer, differences in normalization strategies (per-window vs global) caused significant variance.
- **Fix**: The training pipeline enforces `normalize_array` globally over the whole trial before windowing, preserving relative amplitudes between windows.
