# 01 Project History

## Phase 1: MatchNet Development
- **Goal**: Replicate and enhance the base Contrastive MatchNet for AAD.
- **Method**: Implemented EEGNet and AudioEncoder with InfoNCE loss.
- **Results**: Achieved ~71% Window Accuracy. Discovered issues with standard contrastive formulations.
- **Lessons Learned**: The margin between similarity scores is the key indicator of model certainty.

## Phase 2: LOSO Evaluation
- **Goal**: Rigorously evaluate the model across subjects.
- **Method**: Leave-One-Subject-Out (LOSO) cross-validation on the DTU dataset.
- **Results**: Established a baseline of 68-72% average accuracy, with high inter-subject variability (some >80%, others <60%).
- **Lessons Learned**: The model struggles fundamentally on certain subjects, indicating a need for either personalization or a confidence-based rejection system.

## Phase 3: Confidence Feature Engineering
- **Goal**: Design features that predict when the MatchNet is likely to fail.
- **Method**: Extracted `margin`, `sim_chosen`, `sim_unchosen`, `rolling_std_margin`, and `trial_consistency` from offline predictions.
- **Results**: Found that temporal stability and margin magnitude strongly separate correct from incorrect predictions.
- **Lessons Learned**: Single-window embeddings contain enough information to judge confidence; we do not need access to raw EEG at the confidence level.

## Phase 4: Confidence Validation
- **Goal**: Train and validate the XGBoost Confidence Model.
- **Method**: Trained an XGBoost classifier on the extracted features.
- **Results**: Achieved high AUROC (~0.75-0.80) for predicting correctness. Selective accuracy curves demonstrated significant performance gains.
- **Lessons Learned**: The two-stage pipeline (Deep Encoder -> Feature Extractor -> XGBoost) is highly effective and computationally cheap for runtime.

## Phase 5: Confidence Audits
- **Goal**: Hostile review and failure analysis of the confidence framework.
- **Method**: Conducted Information Gap Audits, Decision Path Audits, SHAP analysis, and ablation studies.
- **Results**: Verified that the confidence model relies heavily on `margin` and `rolling_std_margin`. Disproved hypotheses about spatial leakage.
- **Lessons Learned**: The confidence model is well-calibrated and correctly identifies regions of signal degradation.

## Phase 6: KUL Transfer Experiments
- **Goal**: Evaluate the zero-shot generalizability of the DTU-trained MatchNet on the independent KUL dataset.
- **Method**: Reconstructed the 28-band Gammatone preprocessing pipeline. Processed KUL S1 data and ran distribution audits.
- **Results**: Exposed significant numerical domain shifts between DTU and KUL. Demonstrated that simple scaling mismatches can destroy zero-shot accuracy.
- **Lessons Learned**: Cross-dataset transfer requires rigorous statistical alignment (Domain Adaptation) before passing tensors to a frozen model.
