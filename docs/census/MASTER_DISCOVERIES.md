# MASTER DISCOVERIES — Every Scientific Insight

> Each discovery listed with its evidence, impact, and source file.

---

## Category 1: Data & Label Discoveries

### D-01: The 50% Accuracy Bug — Label Semantics
- **Discovery**: DTU event labels (1/2) encode speaker **gender** (male/female), NOT stream identity (A/B). `wavA` is ALWAYS the attended stream in the preprocessed data.
- **Impact**: 3 days of debugging. Evaluation logic was comparing predictions against gender labels → pure chance (50.09%).
- **Fix**: `correct = 1 if prediction == 'A' else 0` (one line).
- **Source**: [PAPER_FOUNDATION_V2.md](file:///C:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/docs/PAPER_FOUNDATION/PAPER_FOUNDATION_V2.md), Section 2.2
- **Lesson**: Never assume dataset labels mean what their variable names suggest. Verify against the original preprocessing code.

### D-02: 100% Stimulus Overlap
- **Discovery**: ALL test stories in the KUL LOSO evaluation are heard during training by other subjects. There is zero novel stimulus content in any test fold.
- **Impact**: Limits claims of zero-shot stimulus generalization. The model may leverage familiarity with the acoustic structure of the stories.
- **Source**: [PHASE_2_FALSIFICATION.md](file:///C:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/docs/PHASE_2_FALSIFICATION.md)
- **Lesson**: Stimulus overlap ≠ data leakage, but it constrains the generalization claims.

### D-03: KUL Track-Ear Swapping
- **Discovery**: In the KUL dataset, track number is NOT fixed to a specific ear. Tracks swap across trials. The `attended_ear` field definitively determines which stream is ground truth.
- **Impact**: Naive implementations that hard-code `stimuli[0]` as left ear will produce wrong labels ~50% of the time.
- **Source**: [KUL_DATASET_AUDIT.md](file:///C:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/docs/KUL_DATASET_AUDIT.md), Section 4

### D-04: Audio Clustering Validates Labels
- **Discovery**: K-means clustering on raw audio features produces `distinct_cluster_rate = 1.0`, perfectly separating the two speakers. Mapping {0:2, 1:1} achieves accuracy 1.0000.
- **Source**: [DATA_ANALYSIS.md](file:///C:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/DATA_ANALYSIS.md)

---

## Category 2: Architecture & Training Discoveries

### D-05: Contrastive > Reconstruction
- **Discovery**: Reconstruction objectives (mapping EEG → acoustic envelope) are mathematically ill-posed for cross-subject generalization. Contrastive objectives (which stream is the brain tracking?) are fundamentally easier and transfer better.
- **Evidence**: TCN, VLAAI-Lite, EEGNet-TCN all converge to ~50–55% LOSO under reconstruction. ContrastiveMatchNet achieves 69% under the same protocol.
- **Source**: [PAPER_FOUNDATION_V2.md](file:///C:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/docs/PAPER_FOUNDATION/PAPER_FOUNDATION_V2.md), Section 5.3

### D-06: Parameter Asymmetry is Correct
- **Discovery**: ContrastiveMatchNet's parameter distribution is 95.4% audio encoder (48,608) vs 4.6% EEG encoder (2,320). This reflects the fundamental information asymmetry — audio is rich and high-dimensional; EEG is sparse, noisy, and benefits from aggressive regularization.
- **Source**: Architecture analysis in Paper Foundation

### D-07: 28-Band Gammatone >> Single Envelope
- **Discovery**: Feeding the full 28-band Gammatone representation (vs. a single broadband envelope) increased accuracy by ~5 percentage points. The 28-band representation preserves spectral structure that the single-band destroys.
- **Impact**: All audio preprocessing must produce 28 bands. Single-band approaches will fail.
- **Source**: [REPOSITORY_MODEL_LINEAGE.md](file:///C:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/docs/REPOSITORY_MODEL_LINEAGE.md)

### D-08: EEGNet ≈ ATCNet Under LOSO
- **Discovery**: ATCNet (with multi-head attention and TCN, ~15k params) did not produce meaningfully higher cross-subject accuracy than EEGNet (2,320 params). The attention mechanism overfits to training subjects' temporal patterns.
- **Impact**: Chose EEGNet for 6× smaller size and equivalent performance.
- **Source**: Paper Foundation, Section 6.2

### D-09: Training/Evaluation Window Mismatch is Beneficial
- **Discovery**: Training with 5s windows (320 samples) with 2s hop increases training data by ~3× through overlap. Evaluating with 10s windows (640 samples, non-overlapping) gives more stable decisions. This deliberate mismatch improves both training efficiency and evaluation reliability.
- **Source**: Paper Foundation, Section 6.4

### D-10: Cosine Training + Pearson Evaluation
- **Discovery**: The training loss uses cosine similarity internally, while evaluation uses Pearson correlation. The two are nearly equivalent for zero-mean vectors (which BatchNorm ensures), but Pearson gives slightly better empirical evaluation performance.
- **Source**: Paper Foundation, Section 6.5

---

## Category 3: Confidence & Uncertainty Discoveries

### D-11: The Geometric Hypothesis
- **Discovery**: The information needed to predict failure is already encoded in the geometric output of ContrastiveMatchNet. When EMG noise drowns the attention signal, z_eeg wanders aimlessly in the 64-D latent space, landing equidistant from both audio embeddings → small margin.
- **Evidence**: Margin bins show monotonic accuracy: 0.00–0.05 → 57.6%, 0.25–0.30 → 100%.
- **Source**: Paper Foundation, Section 7.3

### D-12: Temporal Features are Essential
- **Discovery**: Margin alone achieves AUROC ~0.65. Adding rolling_std_margin and trial_consistency boosts to ~0.80. Biological artifacts are temporally correlated — a single EMG spike corrupts a cluster of windows, not just one.
- **Source**: Minimal Model Audit (step_5_2)

### D-13: The Information Limit
- **Discovery**: High-confidence failure prediction (predicting when the system will be wrong despite high confidence) saturates at AUROC ≈ 0.59. This is a fundamental theoretical boundary — similarity-derived features cannot reliably predict their own blind spots.
- **Impact**: Future confidence improvements MUST use fundamentally different information sources (e.g., Evidential DL, spectral analysis).
- **Source**: Root Cause Audit (step_5_5), Paper Foundation Section 1.3

### D-14: Failures = EMG Overwrites
- **Discovery**: Low-confidence windows perfectly correspond to massive spikes in broadband EEG power (1–20 Hz). These are the textbook signature of electromyography (EMG) — swallowing, jaw clenching, blinks. The neural signal doesn't degrade; it is **overwritten**.
- **Source**: Root Cause & Information Gap Audit (step_5_5)

### D-15: Confidence Head Shortcut Learning
- **Discovery**: The learned confidence head in Phase 13 was essentially learning `sigmoid(margin)` — a smooth scaling function, not true uncertainty estimation. 46.8% of neurons were dead. The 64-D z_pool was ignored because 1-D margin scalars provide a faster gradient path.
- **Impact**: Motivated redesign to Evidential Deep Learning (EDL) head.
- **Source**: [phase13_confidence_architecture_review.md](file:///C:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/docs/reviews/phase13_confidence_architecture_review.md)

---

## Category 4: Cross-Dataset & Generalization Discoveries

### D-16: Cross-Dataset Latent Alignment
- **Discovery**: DTU and KUL EEG representations occupy an overlapping latent manifold. L2 norms align perfectly between datasets. Silhouette scores confirm no embedding collapse on unseen data.
- **Source**: Phase 10 Independent Audit

### D-17: Aggregation Method Determines Accuracy
- **Discovery**: The 68.24% vs 54.26% cross-dataset accuracy discrepancy is NOT a model error. It is purely a function of trial aggregation: Accumulated Pearson (DTU protocol) vs Majority Vote (KUL protocol). Window predictions, forward passes, and Pearson values are all identical.
- **Impact**: Must always report both metrics. Literature comparison requires protocol awareness.
- **Source**: [phase10_cross_dataset_evaluation.md](file:///C:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/docs/reports/phase10_cross_dataset_evaluation.md)

### D-18: Audio Preprocessing Determines Generalization
- **Discovery**: Initial KUL zero-shot attempts failed entirely (~50%) because simplified single-band envelopes were used. Once the 28-band Gammatone was correctly reconstructed, accuracy jumped to 75.8% (30s windows). The bottleneck was mechanical preprocessing, not the neural network.
- **Source**: [ULTIMATE_PROJECT_ARCHIVE.md](file:///C:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/docs/PAPER_FOUNDATION/ULTIMATE_PROJECT_ARCHIVE.md), Section 7.5

---

## Category 5: Product & Decision Engine Discoveries

### D-19: Window Voting Fails at ~53% Window Accuracy
- **Discovery**: Phase 11 selective AAD using majority vote over 10s windows yielded 0.0% trial accuracy on KUL despite ~53% window accuracy. When half of windows are wrong, majority voting across a trial is pure coin-flip.
- **Resolution**: Temporal evidence accumulation (SPRT/LLR) replaces naive voting.
- **Source**: [phase11_project_state.md](file:///C:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/docs/reports/phase11_project_state.md)

### D-20: Phase 17.2 NaN Equality Bug
- **Discovery**: `np.nan != np.nan` evaluates to `True` in pandas, causing continuous uncertainty periods (output=NaN→NaN) to be counted as false switches. This inflated the False Switch Rate dramatically.
- **Fix**: Explicit NaN-aware equality: `pd.isna(prev) and pd.isna(curr)`.
- **Source**: [phase17_2_report.md](file:///C:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/results/phase17_2/phase17_2_report.md)

### D-21: False Switches are Model Errors, Not Policy Errors
- **Discovery**: The 2 false switches in Phase 17.2 were caused by the Conformer emitting overwhelmingly strong incorrect predictions (margins -0.282 and -0.460). The policy engine correctly obeyed. These are neural network decoding failures, not FSM logic bugs.
- **Source**: [manual_case_studies.md](file:///C:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/results/phase17_2/manual_case_studies.md)

### D-22: Internal vs Output Events
- **Discovery**: Phase 17.2 metrics counted internal controller transitions (every 46.875ms). Phase 17.3 revealed these don't represent what a user hears — many "switches" are sub-second internal oscillations that never reach the audio output. True UX metrics require an Output State Machine that collapses to 1-second stable states.
- **Source**: Phase 17.3 analysis output
