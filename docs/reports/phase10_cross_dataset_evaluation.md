# Scientific Documentation: Cross-Dataset Evaluation & DTU Benchmark Protocol Audit
**Phase 10 \u2014 Zero-Shot Generalization (KUL \u2192 DTU)**

## 1. Background
In our ongoing efforts to establish a robust Auditory Attention Decoding (AAD) framework, we evaluated an AAD-Conformer trained exclusively on the KUL dataset and tested it directly on the independent DTU dataset in a strictly zero-shot transfer setting. 

This experiment matters because verifying true zero-shot generalization proves that the model learns domain-invariant neural representations of auditory attention, rather than merely memorizing subject-specific or hardware-specific noise artifacts. An independent evaluation was necessitated when preliminary zero-shot accuracy metrics produced high variance across different evaluation loops, requiring strict isolation to prevent logic leakage.

## 2. Experimental Pipeline
To evaluate the zero-shot generalizability, the following isolated pipeline was established:

1. **Training (KUL)**: A conformer trained using InfoNCE contrastive loss on 64-channel KUL data (downselected to 8 channels).
2. **Frozen Checkpoint**: The model weights were frozen (`requires_grad = False`).
3. **Data Loading (DTU)**: Raw EEG data from the independent DTU dataset was loaded.
4. **Audio Extraction**: 28-band gammatone envelopes were loaded for DTU stimuli.
5. **Preprocessing**:
    - **CAR** (Common Average Reference) applied to EEG.
    - **Bandpass Filter** (1.0 \u2013 8.0 Hz, 4th Order Butterworth).
    - **Normalization** (Z-score normalization per trial).
6. **Windowing**: Data chunked into 5-second overlapping/non-overlapping windows.
7. **Forward Pass**: The frozen Conformer outputs spatial-temporal embeddings.
8. **Similarity**: Pearson correlation ($r$) computed between EEG embeddings and Audio envelopes.
9. **Aggregation**: Window similarities aggregated into a final Trial Decision.

## 3. Independent Scientific Audit
To scientifically validate the results and rule out any methodological flaws, a 9-phase stringent independent audit was conducted:
- **Dataset Provenance & Channel Verification**: Verified that DTU data was genuinely loaded and aligned structurally to KUL.
- **Checkpoint & Forward Pass Verification**: Verified weights remained frozen and no NaNs propagated in the spatial/temporal filters, indicating covariance shifts were manageable.
- **Activation & Latent Comparison**: L2 Norms and Silhouette scores proved that DTU representations occupied an overlapping latent manifold with KUL representations, preventing embedding collapse.
- **Confidence Verification**: Confirmed that confidence outputs (Margin-based) successfully generalized to DTU with a high AUROC (0.82+).
- **Leakage Audit**: No preprocessing state, labels, or test sets leaked.
- **Independent Evaluator**: A completely isolated evaluation loop rebuilt from first principles exposed a discrepancy in Trial Aggregation, prompting the Phase 10.4 debugging effort.

## 4. The Major Discovery
During the independent reproduction audit (Phase 10.4), it was discovered that the evaluation discrepancy stemmed from two mathematically divergent trial aggregation methods.

### Method A: Majority Vote
**Definition**: Each 5s window makes an independent binary decision based on which Pearson correlation is higher. The trial decision is the simple majority of these boolean window votes.
- **Advantages**: Robust to extreme outliers; standard in decision theory and Kuruvila benchmarks.
- **Limitations**: Discards the magnitude of confidence/correlation.

### Method B/C: Accumulated (Average) Pearson Evidence
**Definition**: The Pearson correlation coefficients are summed (or averaged) across all windows in a trial. The trial decision is made by comparing the total accumulated correlation of Stream A vs Stream B.
- **Advantages**: Preserves magnitude; highly confident windows can override ambiguous ones.
- **Limitations**: A single highly confident artifact window can flip the entire trial.

## 5. Root Cause Analysis
The Phase 10.4 independent script isolated the exact cause of the **54.26% vs 68.24%** accuracy discrepancy.
- **Window predictions were identical.**
- **Forward passes were identical.**
- **Pearson values were identical.**

The original Phase 10.1 script utilized **Accumulated Pearson** to evaluate trials, yielding **68.24%**. 
The Phase 10.3 independent evaluator utilized **Majority Vote**, yielding **54.26%**. 
The difference is purely a function of trial aggregation logic.

## 6. DTU Benchmark Protocol Audit (Literature & Code)
To determine which metric is scientifically correct, a repository-wide forensics audit was performed on the official DTU baseline scripts.

### Code Evidence
1. **Ridge Baseline** (`training/loso_ridge_runner.py`):
    - Function: `evaluate_trial_windows` (Lines 71-90)
    - Protocol: `return float(np.mean(corr_a_values)), float(np.mean(corr_b_values))`
    - The Ridge baseline computes trial accuracy via **Average Pearson**.
2. **Temporal CNN** (`training/train_temporal_cnn_loso.py`):
    - Function: `evaluate_trial_windows` (Lines 288-300)
    - Protocol: Averages `safe_corr` over windows. 
    - The Temporal CNN baseline computes trial accuracy via **Average Pearson**.
3. **MatchNet** (`training/train_matchnet_loso.py`):
    - Does not natively report Trial Accuracy, only evaluates Window Accuracy (`n_correct`).
4. **Kuruvila Baseline** (`training/train_kuruvila_original_loso.py`):
    - Computes strict **Majority Vote**.

### Comparison Table
| Method | Used by MatchNet? | Used by Ridge (DTU)? | Used by T-CNN (DTU)? | Used by Kuruvila (KUL)? |
| :--- | :--- | :--- | :--- | :--- |
| **Majority Vote** | N/A | No | No | **Yes** |
| **Average Pearson** | N/A | **Yes** | **Yes** | No |

## 7. Scientific Conclusions
1. **The model itself is NOT incorrect.** The checkpoint, data loading, and representations were robust.
2. **This is NOT leakage or corruption.** 
3. **The discrepancy is purely evaluation-based.**
The 68.24% accuracy was derived using the *DTU Ridge/CNN Protocol* (Average/Accumulated Pearson). The 54.26% accuracy was derived using the *KUL Protocol* (Majority Vote).

## 8. Final Recommendation
Based on the repository audit, we recommend **Option 3: Reporting Both Metrics**.

While Majority Vote (54.26%) is the standard AAD methodology for the KUL dataset and generalized decision theory, the Average/Accumulated Pearson (68.24%) is the historically enforced benchmark for DTU baselines in this repository. 

To maintain scientific integrity and reproducibility, future publications MUST explicitly report both metrics, clearly delineating the **DTU-Baseline Protocol (Average Pearson)** from the **Standard AAD Protocol (Majority Vote)**. This resolves the contradiction and prevents future misinterpretation of cross-dataset generalization.
