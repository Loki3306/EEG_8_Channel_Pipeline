# Phase 1 — Confidence Benchmarking

## Purpose

Before building any new model, answer one fundamental question:

> Does MatchNet already know when it is right and when it is wrong?

If the answer is **No**, then:

* Selective AAD fails
* Confidence heads become questionable
* Subject-aware confidence becomes much harder

If the answer is **Yes**, then confidence becomes a valid research direction.

This phase exists solely to validate that assumption.

---

# Scientific Question

Current MatchNet outputs:

```python
sim_A
sim_B
```

and prediction:

```python
pred = argmax(sim_A, sim_B)
```

But internally the model may already contain a reliability signal.

Example:

```python
sim_A = 0.95
sim_B = 0.10
```

Very clear decision.

vs

```python
sim_A = 0.52
sim_B = 0.49
```

Very ambiguous decision.

The hypothesis:

> Larger separation between attended and unattended similarity implies higher probability of correctness.

---

# Research Questions

### RQ1
Can MatchNet confidence predict correctness?

### RQ2
Do incorrect predictions have systematically lower confidence?

### RQ3
Are weak LOSO subjects also low-confidence subjects?

### RQ4
Can confidence explain part of the subject variability problem? 

---

# Phase Structure

```text
Step 1
Export raw MatchNet outputs
        ↓

Step 2
Generate confidence scores
        ↓

Step 3
Analyze confidence distributions
        ↓

Step 4
Measure confidence-correctness relationship
        ↓

Decision:
Confidence useful?
```

---

# Step 1 — Prediction Export Framework

## Objective

Modify evaluation pipeline only.
No retraining.
No architecture changes.

---

## Current Output

Today you likely save:

```python
prediction
label
accuracy
```

---

## New Output

For every window save:

```python
subject_id
trial_id
window_id

sim_A
sim_B

prediction
label

correct
```

---

## Example CSV

| Subject | Trial | Window | sim_A | sim_B | Pred | Label | Correct |
| ------- | ----- | ------ | ----- | ----- | ---- | ----- | ------- |
| S7      | 12    | 4      | 0.88  | 0.22  | A    | A     | 1       |
| S7      | 12    | 5      | 0.54  | 0.49  | A    | B     | 0       |

---

## Deliverable

```text
matchnet_predictions.csv
```

This file becomes the foundation of the entire confidence project.

---

# Step 2 — Construct Confidence Scores

We are not yet inventing confidence.
We are extracting confidence already hidden in MatchNet.

---

# Method A — Similarity Margin

## Definition

```python
margin = abs(sim_A - sim_B)
```

---

## Interpretation

Small margin:
```text
Model uncertain
```

Large margin:
```text
Model confident
```

---

## Example

```python
sim_A = 0.90
sim_B = 0.10

margin = 0.80
```
Very confident.

---

```python
sim_A = 0.53
sim_B = 0.49

margin = 0.04
```
Very uncertain.

---

## Why This Matters

This is the most direct confidence signal available inside MatchNet.
No assumptions.
No calibration.
No extra models.

---

# Method B — Normalized Margin

Raw margins may vary.
Normalize:

```python
confidence =
|sim_A - sim_B|
/
(|sim_A| + |sim_B|)
```

Range:

```text
0 → uncertain
1 → certain
```

---

## Goal

Determine whether normalization improves confidence quality.

---

# Method C — Softmax Confidence

Treat similarities as logits.

```python
p = softmax([sim_A, sim_B])

confidence = max(p)
```

---

## Example

```python
[0.9,0.1]
```
↓
```python
[0.69,0.31]
```

Confidence:
```python
0.69
```

---

## Why Include It

Not because it is good.
Because reviewers will expect it.
This becomes the baseline confidence method.

---

# Step 3 — Confidence Distribution Analysis

Before computing metrics:
Look at distributions.

---

# Analysis A

Correct vs Incorrect

Create:
```text
Correct Predictions
Incorrect Predictions
```
confidence histograms.

---

Desired Finding

```text
Correct:
higher confidence

Incorrect:
lower confidence
```

---

## Visualization

Histogram
```text
Confidence
```
for:
```text
Correct
Incorrect
```
overlaid.

---

# Analysis B

Subject-wise Confidence

Compute:
```python
mean_confidence(subject)
```
for every subject.

---

Example

| Subject | Accuracy | Mean Confidence |
| ------- | -------- | --------------- |
| S7      | 81%      | 0.76            |
| S15     | 79%      | 0.74            |
| S10     | 56%      | 0.51            |
| S11     | 56%      | 0.49            |

---

Question:

> Do weak subjects naturally generate lower confidence?

This is the first bridge toward Subject-Aware Confidence.

---

# Analysis C

Trial-wise Confidence

For every trial:
```python
trial_confidence
trial_accuracy
```

---

Question:
Can confidence identify difficult trials?

---

# Step 4 — Confidence Binning

This is the most important experiment.

---

## Procedure

Sort predictions by confidence.

Create bins:
```text
0.0-0.1
0.1-0.2
0.2-0.3
...
0.9-1.0
```

---

For each bin:
Compute:
```python
accuracy
```

---

Example

| Confidence Bin | Accuracy |
| -------------- | -------- |
| 0.0-0.1        | 52%      |
| 0.1-0.2        | 58%      |
| 0.2-0.3        | 64%      |
| 0.3-0.4        | 72%      |
| 0.4-0.5        | 81%      |
| >0.5           | 92%      |

---

## Desired Result

Monotonic increase.

```text
Higher Confidence
↓
Higher Accuracy
```

---

# Statistical Tests

## Spearman Correlation

Between:
```python
confidence
correctness
```

---

Measure:
```python
rho
p_value
```

---

Desired
```text
rho > 0
p < 0.05
```

---

## Point-Biserial Correlation

Since correctness is binary:
```text
Correct = 1
Wrong = 0
```
also compute:
```python
point_biserial_r
```

---

# Success Criteria

Phase 1 is considered successful if:

### Criterion 1
Confidence distributions separate:
```text
Correct
vs
Incorrect
```

---

### Criterion 2
Confidence bins show monotonic accuracy growth.

---

### Criterion 3
Confidence-correctness correlation is statistically significant.

---

### Criterion 4
Weak subjects exhibit lower confidence than strong subjects.

This is especially important because it supports the larger hypothesis that subject variability is the core bottleneck rather than audio representation. 

---

# Deliverables

At the end of Phase 1 you should have:

### Data
```text
matchnet_predictions.csv
matchnet_confidence.csv
```

### Figures
```text
Figure 1: Correct vs Incorrect confidence histogram
Figure 2: Subject-wise confidence
Figure 3: Confidence bins vs Accuracy
Figure 4: Confidence vs Correctness scatter
```

### Statistics
```text
Spearman correlation
Point-biserial correlation
Bin-wise accuracy table
Subject-wise confidence table
```

---

# Decision Gate

### If Phase 1 Fails
Confidence is not predictive.
Stop the confidence roadmap.
Return to subject adaptation research.

---

### If Phase 1 Succeeds
Proceed to:
```text
Phase 2: Reliability Analysis
(AUROC, ECE, Brier, Reliability Diagrams)
```
Only after Phase 2 succeeds should we move toward Selective AAD and eventually Subject-Aware Confidence.
