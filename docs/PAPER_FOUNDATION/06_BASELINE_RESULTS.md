# 06 Baseline Results

## MatchNet LOSO Results (DTU Dataset)
*Note: Results reflect the primary DTU evaluation run across 18 subjects.*

| Metric | Value |
|--------|-------|
| Overall Window Accuracy | ~71.0% |
| Overall Trial Accuracy | ~85.0% |

### Subject-Wise Window Accuracy (Sample)
*Significant inter-subject variability was observed.*

| Subject | Accuracy |
|---------|----------|
| S1      | 76.2%    |
| S2      | 81.4%    |
| S3      | 58.1%    |
| S4      | 72.8%    |
| ...     | ...      |

## Ridge Regression Baseline (Reference)
*Note: Ridge AAD was used as the historical reference model.*

| Metric | Value |
|--------|-------|
| Overall Window Accuracy | ~65.0% |
| Overall Trial Accuracy | ~78.0% |

## Window-Length Ablation (KUL Generalization)
*Note: Zero-shot transfer results from KUL S1.*

| Window Length | Window Accuracy | Trial Accuracy | Confidence AUROC |
|---------------|-----------------|----------------|------------------|
| 30s           | 75.8%           | 100.0%         | NaN (all correct)|
| 20s           | 71.4%           | 95.0%          | 0.952            |
| 15s           | 69.1%           | 95.0%          | 0.814            |
| 10s           | 66.8%           | 90.0%          | 0.812            |
| 5s            | 60.1%           | 90.0%          | 0.729            |
| 2s            | 54.3%           | 75.0%          | 0.612            |
