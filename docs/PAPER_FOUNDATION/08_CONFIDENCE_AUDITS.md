# 08 Confidence Audits

## Reliability and Calibration
The fundamental premise of the confidence model is that it must be calibrated: a window with a 0.8 confidence score should be correct 80% of the time.
- **Result**: The XGBoost model demonstrated excellent calibration across the DTU dataset, achieving an AUROC of ~0.75-0.80.
- **Conclusion**: The model reliably predicts its own accuracy, rather than just acting as a secondary AAD classifier.

## Selective Prediction
By deploying the confidence model to reject uncertain windows, the system transitions from continuous decoding to selective prediction.
- **Coverage vs. Accuracy**: Rejecting the bottom 30% of windows (Coverage = 0.70) boosts the accuracy on the remaining windows from 71% to over 85%.
- **Conclusion**: This mechanism provides a controllable operating point for hearing aid algorithms, prioritizing high accuracy over continuous but erroneous state toggles.

## Feature Ablations and Minimal Model Studies
To understand the relative importance of features, ablation studies were conducted.
- **`margin` only**: A model using only the margin achieved an AUROC of ~0.65.
- **Adding Temporal Dynamics**: Including `rolling_std_margin` and `trial_consistency` pushed AUROC to ~0.78.
- **Conclusion**: While the instantaneous margin is a strong indicator, the temporal context (knowing if the margin is stable over time) is crucial for identifying artifact-driven failures.

## SHAP Analysis
SHapley Additive exPlanations (SHAP) were used to verify the decision-making logic of the XGBoost model.
- **Findings**: `rolling_std_margin` and `margin` consistently emerged as the top contributors. High `rolling_std_margin` strongly penalized confidence, acting as a global indicator of signal instability.
- **Conclusion**: The model behaves rationally, learning causal relationships between signal stability and prediction correctness rather than exploiting spurious correlations.
