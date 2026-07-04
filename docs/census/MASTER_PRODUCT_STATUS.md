# MASTER PRODUCT STATUS — Hearing-Aid Platform State

> Current state of the complete streaming hearing-aid AAD pipeline.

---

## 1. System Architecture (Current)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    HEARING AID AAD PIPELINE                              │
│                                                                          │
│  ┌─────────────┐    ┌──────────────────┐    ┌───────────────────────┐   │
│  │  EEG Sensor  │───→│  AAD-Conformer   │───→│  Confidence Head      │   │
│  │  8 channels  │    │  (FROZEN)        │    │  (Late-Fusion MLP)    │   │
│  │  64 Hz       │    │  ~2M params      │    │  AUROC 0.7337         │   │
│  └─────────────┘    └──────────────────┘    └──────────┬────────────┘   │
│                                                         │                │
│  ┌─────────────┐    ┌──────────────────┐               │                │
│  │  Audio A     │───→│  Audio Encoder   │               │                │
│  │  28-band     │    │  (shared weights)│───┐           │                │
│  └─────────────┘    └──────────────────┘   │           │                │
│                                             │           │                │
│  ┌─────────────┐    ┌──────────────────┐   │           │                │
│  │  Audio B     │───→│  Audio Encoder   │───┤  Pearson  │                │
│  │  28-band     │    │  (shared weights)│   │  Corr     │                │
│  └─────────────┘    └──────────────────┘   │           │                │
│                                             ▼           ▼                │
│                              ┌──────────────────────────────┐           │
│                              │  SequentialWindowBuffer       │           │
│                              │  (Temporal Memory)            │           │
│                              └──────────────┬───────────────┘           │
│                                              │                           │
│                              ┌──────────────▼───────────────┐           │
│                              │  ContextAwarePolicyEngine     │           │
│                              │  (FSM with 5 heuristics)     │           │
│                              │  States: INIT → WAIT → LOCK  │           │
│                              │          → STABILIZE → SWITCH │           │
│                              │          → COOLDOWN → UNCERTAIN│          │
│                              └──────────────┬───────────────┘           │
│                                              │                           │
│                              ┌──────────────▼───────────────┐           │
│                              │  HEARING AID OUTPUT           │           │
│                              │  Beamformer: Speaker A or B   │           │
│                              │  Or: COAST (maintain lock)    │           │
│                              └──────────────────────────────┘           │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Pipeline Components & Status

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| EEGNet Encoder | [eegnet.py](file:///C:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/models/eegnet.py) | ✅ FROZEN | 2,320 params |
| AudioEncoder | [matchnet.py](file:///C:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/models/matchnet.py) | ✅ FROZEN | 48,608 params |
| ContrastiveMatchNet | [matchnet.py](file:///C:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/models/matchnet.py) | ✅ FROZEN | 50,928 params total |
| AAD-Conformer | [aad_conformer.py](file:///C:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/models/aad_conformer.py) | ✅ FROZEN | ~2M params |
| Confidence Head | In aad_conformer.py | ⚠️ NEEDS REDESIGN | Compressed dynamic range |
| XGBoost Confidence | Trained artifact | ✅ ACTIVE | AUROC 0.8057 (DTU) |
| WindowBuffer | [window_buffer.py](file:///C:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/decision_engine/window_buffer.py) | ✅ ACTIVE | Passive memory |
| EvidenceAccumulator | [window_buffer.py](file:///C:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/decision_engine/window_buffer.py) | ✅ ACTIVE | Dirichlet-based |
| DecisionPolicyEngine v1 | [decision_policy_engine.py](file:///C:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/decision_policy_engine.py) | ✅ ACTIVE | 5-state FSM |
| ContextAwarePolicyEngine v2 | [context_aware_engine.py](file:///C:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/decision_engine/context_aware_engine.py) | ✅ ACTIVE | 7-state FSM + 5 heuristics |
| Hardware Emulator | [phase16_continuous_simulator.py](file:///C:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/analysis/phase16_continuous_simulator.py) | ✅ ACTIVE | 5 scenarios |
| Scenario Definitions | scenarios/*.json | ✅ ACTIVE | 5 reusable scenarios |
| KUL Adapter | In phase16 | ✅ ACTIVE | ContinuousSessionGenerator |
| DTU Adapter | Planned | ❌ NOT STARTED | — |

---

## 3. Decision Engine Configuration (Current)

### ContextAwarePolicyEngine Defaults

| Parameter | Value | Purpose |
|-----------|-------|---------|
| base_threshold | 0.85 | Confidence needed to lock |
| minimum_lock_duration | 5 windows | Ignore fluctuations after locking |
| minimum_switch_gap | 10 windows | Minimum time between switches |
| minimum_consecutive_windows | 3 | Consecutive agreements needed |
| maximum_wait_time | 15 windows | Timeout before UNCERTAIN |
| uncertainty_threshold | 0.15 | Band around 0.5 for uncertainty |
| cooldown_duration | 15 windows | Post-switch refractory period |
| stabilizing_threshold | 30 windows | Windows in LOCKED before STABILIZING |

### Active Heuristics

| Heuristic | Effect |
|-----------|--------|
| difficulty | Adjusts threshold ±0.10–0.15 based on first 5 windows |
| growth_rate | Adjusts consecutive requirement based on evidence slope |
| oscillation_penalty | +10 switch gap, +0.05 threshold if ≥2 recent switches |
| hysteresis | +0.05 threshold, +2 consecutive when STABILIZING |
| cooldown | Post-switch refractory period (ignores counter-evidence) |

---

## 4. Current UX Metrics (Phase 17.3)

### What the User Experiences

| Metric | Value | Interpretation |
|--------|-------|---------------|
| **Audible False Switches/hr** | 22.53 | User hears ~23 wrong switches per hour |
| **Decision Availability** | 99.63% | System has an answer 99.6% of the time |
| **Correct Lock Coverage** | 84.48% | Locked onto correct speaker 84.5% of session |
| **Acquisition Latency** | 4.99s | ~5 seconds to first lock after session start |
| **Switch/Recovery Latency** | 25.41s | ~25 seconds to re-lock after a scene change |

### Assessment

| Metric | Target (Hearing Aid Grade) | Current | Gap |
|--------|---------------------------|---------|-----|
| False Switches/hr | < 5 | 22.53 | **4.5× over target** |
| Decision Availability | > 95% | 99.63% | ✅ EXCEEDS |
| Correct Lock Coverage | > 90% | 84.48% | **-5.5 pp** |
| Acquisition Latency | < 3s | 4.99s | **+2s** |
| Switch Latency | < 10s | 25.41s | **2.5× over target** |

---

## 5. Verified Phase 17.2 Controller Metrics

| Metric | Value |
|--------|-------|
| True Switches (across 5 scenarios) | 2 |
| False Switches (across 5 scenarios) | 2 |
| Switch Precision | 50.0% |
| Correct Lock Coverage | 92.58% |
| Total Windows Evaluated | ~109,000 |

### Case Studies

**Case 1 (True Switch)**: Scenario 4, t=24.81s, margin +0.502, correctly switched to Speaker A
**Case 2 (False Switch)**: Scenario 4, t=70.70s, margin -0.282, model confidently wrong → policy obeyed
**Case 3 (False Switch)**: Scenario 5, t=1560.39s, margin -0.460, model overwhelmingly wrong → policy obeyed

**Root Cause of All False Switches**: Neural network decoding errors, NOT policy engine bugs. The Conformer emitted strong incorrect predictions.

---

## 6. Hardware Emulator Status

### Scenarios Available

| # | Name | Duration | Purpose | Status |
|---|------|----------|---------|--------|
| 1 | stable_conversation | ~50s | Baseline stability | ✅ Validated |
| 2 | single_shift | ~180s | A→B→A switch latency | ✅ Validated |
| 3 | rapid_conversation | ~150s | Rapid A→B→A→B stress test | ✅ Validated |
| 4 | mixed_difficulty | ~200s | Multiple concatenated trials | ✅ Validated |
| 5 | long_continuous | ~1800s | Multi-hour session simulation | ✅ Validated |

### Streaming Specifications
- **Window Size**: 2 seconds
- **Hop**: 50ms
- **Sampling Rate**: 64 Hz
- **Boundary Handling**: Windows span scene boundaries (center-sample convention)
- **Ground Truth**: Flips at exact splice timestamps defined in JSON

---

## 7. Deployment Readiness Assessment

### Ready for Deployment
- [x] Neural network trained and validated (LOSO, multi-seed, falsification)
- [x] Confidence estimation operational (AUROC 0.73–0.81)
- [x] Complete streaming pipeline implemented
- [x] Context-aware policy engine with 5 heuristics
- [x] Hardware emulator with 5 reproducible scenarios
- [x] Cross-dataset generalization proven (KUL→DTU zero-shot)
- [x] Product metrics redesigned for user-perceived evaluation

### NOT Ready for Deployment
- [ ] False switch rate 4.5× too high (22.53/hr vs <5 target)
- [ ] Switch latency 2.5× too slow (25.41s vs <10s target)
- [ ] Confidence head needs Evidential DL redesign (compressed dynamic range)
- [ ] No dry-electrode hardware validation
- [ ] No hearing-impaired subject validation
- [ ] No real-time inference benchmarking on target DSP
- [ ] No INT8 quantization tested
- [ ] DTU adapter not implemented for hardware emulator
- [ ] No multi-speaker (>2) support
