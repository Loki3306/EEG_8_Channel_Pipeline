# Phase 12.1 Scientific Report: Sequential Window Buffer

## 1. Purpose
The objective of Phase 12.1 is to establish a robust, model-agnostic infrastructure for temporal memory within the Confidence-Aware Sequential Decision System. This phase introduces the **Sequential Window Buffer** to store window-level predictions, confidences, margins, and Pearson correlations chronologically.

## 2. Architecture and Pipeline Placement
The buffer sits immediately downstream of the frozen AAD-Conformer inference pipeline.

**Current Pipeline Integration:**
```text
EEG -> Frozen Conformer -> Prediction Extraction -> Sequential Window Buffer
```

The buffer passively consumes predictions without modifying them. It is built strictly as memory infrastructure, encapsulating all window information in a `WindowPrediction` dataclass and supporting running statistics (e.g., `running_mean_confidence`, `running_accuracy`) and history retrieval.

## 3. Design Rationale
- **Isolation from Decision Logic:** The buffer is explicitly devoid of decision-making capabilities (e.g., majority vote, evidence accumulation, WAIT states). This isolation ensures that the foundational memory mechanism is stable and independently verifiable before any complex sequential policies are layered on top.
- **Model Agnostic:** The buffer relies solely on the outputs of the inference engine (predictions, margins, confidence probabilities), allowing the underlying AAD-Conformer to be updated or swapped without affecting the decision engine's memory.
- **Detailed Tracing:** The infrastructure supports extremely verbose per-window and per-trial logging, generating a canonical `buffer_trace.csv`. This trace is critical for offline analysis and the development of future decision policies.

## 4. Scientific Limitations
- **No Expected Accuracy Improvements:** As this phase implements pure memory infrastructure without an active decision policy, it does not alter the underlying predictive performance of the AAD-Conformer.
- **No Thresholding or Voting:** It deliberately avoids implementing any threshold logic or naive voting mechanisms (which were shown in Phase 11 to fail spectacularly under near-chance window accuracies).

## 5. Future Dependencies
This infrastructure is a prerequisite for all subsequent stages of Phase 12:
- **Stage 2 (Prediction Stability):** Analyzing how confidence and margins fluctuate across sequential windows.
- **Stage 3 (Evidence Accumulation):** Summing Pearson margins or confidence scores over time to build robust trial-level decisions.
- **Stage 4 (Adaptive Threshold):** Dynamically adjusting acceptance thresholds based on running confidence statistics.
- **Stage 5 (WAIT State):** Allowing the system to defer decisions during periods of low confidence.
