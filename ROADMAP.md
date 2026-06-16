# EEG Auditory Attention Decoding (AAD): Research Roadmap

## 1. Executive Summary
This document outlines the strategic research roadmap for building a practical, low-channel EEG Auditory Attention Decoding (AAD) system. The core objective is to determine which of two competing speakers a user is attending to, targeting neuro-steered hearing aids and wearable EEG applications in real-time. 

Recent experiments demonstrate that cross-subject generalization remains the paramount bottleneck. Short window decoding (<5s) also remains heavily dominated by decision noise. Moving forward, the research agenda explicitly halts open-ended exploratory diagnostics and pivots decisively toward rigorous, publication-driven milestones: evaluating window length evidence accumulation, conducting a calibration study, and optimizing short window performance.

## 2. Core Research Directives & Rules
**Rule 1: No new analysis unless it directly supports:**
1. Accuracy improvement
2. Generalization improvement
3. Calibration improvement
4. Publication-quality figures/tables

**Rule 2: Halt exploratory diagnostic work.**
Current evidence confirms:
- Subject variability exists.
- Domain shift is moderately supported.
- Signal quality is moderately supported.
- Neither explanation strictly dominates.
Additional PCA/UMAP/outlier analyses have diminishing scientific value and will not be pursued unless directly tied to the primary milestones below.

---

## 3. Detailed Roadmap Milestones

### Milestone 1: Window Length Benchmark
**Motivation:** Determine the empirical limits of evidence accumulation. Can we operate in real time?
**Research Question:** How much evidence accumulation is required for reliable 8-channel LOSO AAD?
**Execution:**
- Implement evaluation across `2s, 5s, 10s, 20s, 30s` windows.
- Run for all LOSO subjects.
**Deliverables (Publication Figures & Tables):**
- Subject-wise accuracy table.
- Mean accuracy table.
- Accuracy vs Window Length figure.
- Recovery curves tracking worst-performing subjects over longer windows.
**Estimated Effort:** 1-2 days.

### Milestone 2: Calibration Study
**Motivation:** Zero-shot cross-subject generalization is extremely hard. A short calibration phase might be highly practical for real-world devices. This is likely the strongest publication contribution.
**Research Question:** Can lightweight calibration close the cross-subject performance gap?
**Execution:**
- Implement subject adaptation/fine-tuning experiments using data slices of `0s, 30s, 1min, 2min, 5min`.
- Measure accuracy gain, gain per minute of calibration, and subject-wise improvement.
**Deliverables (Publication Figures & Tables):**
- Calibration curve (Accuracy vs Calibration Time).
- Calibration efficiency figure.
- Best/worst subject response analysis.
**Estimated Effort:** 2-4 days.

### Milestone 3: Short Window Optimization
**Motivation:** State-of-the-art models already do well on 10s windows. Sub-5-second performance carries much stronger scientific novelty for wearable/real-time applications.
**Research Question:** Can 8-channel wearable EEG decode attention within 2-5 seconds?
**Execution:**
- Focus entirely on improving `2s` and `5s` performance.
- Evaluate potential domain adaptation algorithms (CORAL, DANN, MMD) ONLY after Milestones 1 & 2 are complete.
**Estimated Effort:** 1-2 weeks.

---

## 4. Explicit Non-Priorities
To prevent analysis paralysis and scope creep, the following approaches are **explicitly halted**:
- Further Outlier Detection (Isolation Forest, Mahalanobis).
- Further Centroid Analysis.
- Further Dimensionality Reduction (PCA, UMAP, t-SNE).
- SHAP or explainability analysis (deferred until Milestones 1 & 2 are complete).
- Advanced/complex audio models (HuBERT, wav2vec) or new dataset acquisition.

## 5. Timeline & Expected Progress
1. **Phase 1 (Immediate):** Milestone 1 (Window Length Benchmark).
2. **Phase 2 (Near-Term):** Milestone 2 (Calibration Study).
3. **Phase 3 (Medium-Term):** Milestone 3 (Short Window Optimization). Domain adaptation algorithms may be introduced here.

By strictly adhering to these milestones, the project will output three distinct, publishable empirical results before engaging in open-ended algorithm design.