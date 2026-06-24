# 12 All Results Tables

## 1. MatchNet LOSO Accuracy (DTU)
*Source: `analysis/step_1_evaluate_checkpoints.py`*

| Subject | Acc | Subject | Acc |
|---------|-----|---------|-----|
| S1      | 76% | S10     | 68% |
| S2      | 81% | S11     | 72% |
| S3      | 58% | S12     | 75% |
| S4      | 72% | S13     | 60% |
| S5      | 69% | S14     | 80% |
| S6      | 70% | S15     | 71% |
| S7      | 77% | S16     | 65% |
| S8      | 64% | S17     | 69% |
| S9      | 83% | S18     | 73% |

*Average*: ~71.0%

## 2. Confidence Feature Importance (SHAP)
*Source: `analysis/step_5_4_decision_path_audit.py`*

| Feature | Importance Weight |
|---------|-------------------|
| `margin` | 0.42 |
| `rolling_std_margin` | 0.35 |
| `sim_chosen` | 0.12 |
| `trial_consistency` | 0.08 |
| `sim_unchosen` | 0.03 |

## 3. Selective Accuracy vs Coverage
*Source: `analysis/step_5_1_behavior_audit.py`*

| Coverage (%) | Selective Accuracy (%) |
|--------------|------------------------|
| 100          | 71.2                   |
| 90           | 75.4                   |
| 80           | 79.1                   |
| 70           | 83.5                   |
| 60           | 86.2                   |
| 50           | 88.9                   |

## 4. KUL S1 Transfer Ablation
*Source: `analysis/step_6_8_kul_ablation_and_confidence.py`*

| Window Length | Window Accuracy | Trial Accuracy | Confidence AUROC |
|---------------|-----------------|----------------|------------------|
| 30s           | 75.8%           | 100.0%         | NaN              |
| 20s           | 71.4%           | 95.0%          | 0.952            |
| 15s           | 69.1%           | 95.0%          | 0.814            |
| 10s           | 66.8%           | 90.0%          | 0.812            |
| 5s            | 60.1%           | 90.0%          | 0.729            |
| 2s            | 54.3%           | 75.0%          | 0.612            |
