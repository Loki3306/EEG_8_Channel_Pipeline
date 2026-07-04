# MASTER ROADMAP — Canonical Research & Product Roadmap

> Grounded in repository evidence. Every item traces back to a specific finding.

---

## Tier 1: Immediate (Next 2 Phases)

### Phase 18: Evidential Confidence Head Implementation
**Priority**: CRITICAL
**Motivation**: Phase 13 diagnosed the current confidence head as broken — compressed dynamic range [0.35, 0.52], 46.8% dead neurons, shortcut learning via margin.
**Design** (from [phase13_confidence_architecture_review.md](file:///C:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/docs/reviews/phase13_confidence_architecture_review.md)):
1. Replace BCE loss with Evidential MSE Loss (Dirichlet parameterization)
2. Input ONLY z_pool (no margin/corr_a/corr_b) to prevent shortcut learning
3. Output: Evidence parameters (e ≥ 0) via Softplus activation
4. Separate aleatoric (data noise) from epistemic (model ignorance) uncertainty
5. Integrate with existing `TemporalEvidenceAccumulator` in window_buffer.py

**Success Criteria**:
- Dynamic range recovery: confidence outputs span [0.05, 0.95]
- AUROC > 0.75 (matching or exceeding current 0.7337)
- OOD detection: Random/Zero EEG → epistemic uncertainty > 0.8
- Dead neuron rate < 10%

**Evidence Trail**: Discovery D-15, Failure F-12, Phase 13 review document

---

### Phase 19: Switch Latency Reduction
**Priority**: HIGH
**Motivation**: Current switch/recovery latency = 25.41s (target < 10s). This makes the hearing aid sluggish during speaker transitions.
**Approach Options**:
1. **Reduce minimum_switch_gap** from 10 to 5 windows (risk: increased oscillation)
2. **Evidence burst detection**: If evidence slope > threshold, bypass consecutive requirement
3. **Asymmetric thresholds**: Lower threshold to break out of current lock vs. threshold to establish new lock
4. **Warm-start evidence**: Pre-seed evidence accumulator with prior from the first strong-signal window

**Constraints**: Must not increase false switch rate (currently 22.53/hr, already 4.5× over target)

**Evidence Trail**: Phase 17.3 metrics, TRANSITION_SEMANTICS.md

---

## Tier 2: Short-Term (3–6 Months)

### Phase 20: False Switch Rate Reduction
**Priority**: HIGH
**Target**: < 5 audible false switches per hour (currently 22.53)
**Approach Options**:
1. **Tighter oscillation penalty**: Increase cooldown_duration, make stabilizing harder to break
2. **Output state machine smoothing**: Require 2+ seconds of consistent new-speaker output before committing
3. **Confidence-gated switching**: Only allow switches when Evidential confidence > 0.90 (current switches trigger at -0.282 to -0.460 margin — the model is confidently wrong)
4. **Per-scene difficulty adaptation**: Use first N windows to calibrate expectations

**Root Cause**: False switches are caused by the neural network emitting strong incorrect predictions (margin -0.282, -0.460). The policy engine correctly obeys. The fix must be at the confidence layer.

### Phase 21: DTU Adapter for Hardware Emulator
**Priority**: MEDIUM
**Scope**: Implement `DTUAdapter(DatasetAdapter)` to enable domain-shift scenarios
**Design** (from [phase16_2_report.md](file:///C:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/results/phase16_2/phase16_2_report.md)):
1. Implement DTUAdapter mirroring KULAdapter
2. Write `domain_shift.json` scenario: Scene 1 = KUL, Scene 2 = DTU
3. Validate generator handles mid-stream dataset context switches

### Phase 22: Window Length Optimization
**Priority**: MEDIUM
**Motivation**: Window scaling table shows accuracy peaks at 10–20s (77.125%) then drops at 60s (75.31%). Current streaming uses 2s windows.
**Questions**:
1. Is there an optimal streaming window length that balances latency and accuracy?
2. Can adaptive window lengths be used (short during transitions, long during stability)?

---

## Tier 3: Medium-Term (6–12 Months)

### Phase 23: Channel Reduction Study
**Priority**: MEDIUM
**Motivation**: Current system uses 8 channels. Commercial hearing aids may support only 2–4.
**Design**:
1. Systematic ablation: 8 → 6 → 4 → 2 channels
2. Channel importance ranking via spatial filter analysis
3. Re-evaluate confidence AUROC under each configuration
4. Identify minimum viable channel count

**Known Data**: Ridge 2ch = 55.19% vs 8ch = ~65%. The degradation slope must be mapped for deep models.

### Phase 24: Attention Switching Dataset
**Priority**: HIGH (for clinical validity)
**Motivation**: DTU and KUL consist of continuous sustained attention trials. Real hearing aid usage involves frequent, deliberate attention switches.
**Requirements**:
1. Acquire or design experiments with explicit cued attention switches
2. Measure true switch detection latency (not proxy from artificial splices)
3. Validate confidence behavior during genuine transition periods

**Evidence Trail**: ULTIMATE_PROJECT_ARCHIVE.md Section 8.4, PAPER_FOUNDATION_V2.md future work

### Phase 25: Multi-Speaker Extension
**Priority**: LOW (research frontier)
**Motivation**: Current system is hardcoded to exactly 2 speakers. Real environments have 3+.
**Requirements**:
1. Reformulate contrastive loss for N-way comparison
2. Redesign margin metric (currently binary sim_A - sim_B)
3. Test on multi-speaker datasets (if available)

**Evidence Trail**: ULTIMATE_PROJECT_ARCHIVE.md Section 8.3

---

## Tier 4: Long-Term (12+ Months)

### Phase 26: DSP Quantization & Edge Deployment
**Priority**: EVENTUAL
**Pathway**:
1. **Phase A (Cloud/Edge)**: Stream EEG via Bluetooth to smartphone, run MatchNet inference on phone
2. **Phase B (On-Device)**: INT8 quantization of EEGNet (2,320 params) + AudioEncoder (48,608 params), deploy to hearing aid DSP
3. **Phase C (Optimized)**: Pruning, knowledge distillation to sub-10K parameter model

**Constraints**: Battery life (target: 12-hour hearing aid day), latency (<100ms inference), memory (<256KB)

### Phase 27: Dry-Electrode Validation
**Priority**: EVENTUAL
**Motivation**: All results use clinical-grade wet electrodes. Dry electrodes have higher impedance and structural noise.
**Requirements**: Partner with hardware team for in-ear/around-ear dry electrode EEG recordings

### Phase 28: Hearing-Impaired Subject Validation
**Priority**: EVENTUAL
**Motivation**: All 34 subjects (18 DTU + 16 KUL) have normal hearing. Hearing loss may alter cortical tracking patterns.

### Phase 29: Online Adaptive Confidence
**Priority**: EVENTUAL
**Design**: Allow confidence thresholds to adapt dynamically based on current environment. Lower threshold in quiet room, higher in crowded restaurant. Requires environmental noise classification.

---

## Research Questions Registry

| # | Question | Status | Phase |
|---|----------|--------|-------|
| Q1 | Can EDL break the AUROC 0.59 information limit? | OPEN | 18 |
| Q2 | What is the minimum viable channel count? | OPEN | 23 |
| Q3 | Can switch latency be reduced below 5s? | OPEN | 19 |
| Q4 | Does confidence generalize to dry electrodes? | OPEN | 27 |
| Q5 | Can the system handle 3+ speakers? | OPEN | 25 |
| Q6 | Does cortical tracking differ in hearing-impaired subjects? | OPEN | 28 |
| Q7 | Is Conformer superiority preserved under quantization? | OPEN | 26 |
| Q8 | Can attention switches be detected in real-time? | OPEN | 24 |
| Q9 | What is the optimal streaming window length? | PARTIALLY ANSWERED | 22 |
| Q10 | Are temporal confidence features universal across datasets? | ANSWERED (yes) | 9 |
| Q11 | Is the margin a universal confidence proxy? | ANSWERED (yes, AUROC 0.66) | 8 |
| Q12 | Does majority vote vs accumulated Pearson matter? | ANSWERED (yes, dramatically) | 10 |

---

## Publication Roadmap

### Paper 1: Selective AAD (Ready to Write)
- **Title**: "Confidence-Aware Selective Auditory Attention Decoding from Low-Density EEG via Contrastive Representation Learning"
- **Foundation**: [PAPER_FOUNDATION_V2.md](file:///C:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/docs/PAPER_FOUNDATION/PAPER_FOUNDATION_V2.md) (complete draft)
- **Key Results**: 69% → 81.55% @ 70% coverage, AUROC 0.8057, 8 hostile audits
- **Status**: Foundation document complete, needs formatting for target venue

### Paper 2: Evidential Confidence (Phase 18 Dependent)
- **Key Contribution**: First application of Evidential Deep Learning to AAD confidence estimation
- **Status**: Design complete (Phase 13 review), implementation pending

### Paper 3: Streaming Controller (Phase 17.3+ Dependent)
- **Key Contribution**: Complete FSM-based streaming controller with UX metrics
- **Status**: Metrics framework implemented, results need optimization (switch rate, latency)
