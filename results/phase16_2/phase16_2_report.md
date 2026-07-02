# Phase 16.2 Continuous Session Generator Report

## 1. Architecture
The architecture abstracts offline dataset structures into a chronologically stable 64Hz stream, emulating a live hardware ring-buffer.
```
[Scenario JSON] -> [Continuous Session Generator]
                           |
                     [Dataset Adapter]
                           |
           [Raw NumPy Array Array Concatenation]
                           |
             [2-second Sliding Window (50ms hop)]
                           |
                 [Stream to Controller]
```

## 2. Session Generator Design
The `ContinuousSessionGenerator` is designed to guarantee zero skipped windows at scene boundaries. It does this by fetching all the raw 64Hz segments of a scenario, stitching them together into a single unified matrix along the time axis, and *then* applying the overlapping sliding window over the continuous matrix. This prevents "seams" and guarantees timestamps remain perfectly sequential.

## 3. Dataset Adapter Design
To decouple the generator from specific datasets, the `DatasetAdapter` base class was introduced. The `KULAdapter` implements this base class, interfacing with the existing `KULCachedLoader`. In the future, a `DTUAdapter` can be seamlessly injected.

## 4. Streaming API
The output of the generator perfectly mimics a live buffer. Calling `generate_stream()` yields chronological dictionaries containing:
- `eeg_window`: (Channels, Samples)
- `audio_a_window`: (Samples,)
- `audio_b_window`: (Samples,)
- `ground_truth`: 0 or 1
- `timestamp_sec`: Absolute time in seconds from session start
- `scene_name`: Current scene context
- `scenario_name`: Current scenario context
- `window_idx`: Sequential integer index

## 5. Scenario Definitions
Five reusable scenarios have been implemented in `scenarios/`:
1. `1_stable_conversation.json`: A single full-length trial to baseline stability and Hysteresis logic.
2. `2_single_shift.json`: A simple `A -> B -> A` shift to isolate and measure switch latency.
3. `3_rapid_conversation.json`: `A -> B -> A -> B` (60s, 30s, 30s, 30s) to stress the Cooldown and Oscillation Penalty logics.
4. `4_mixed_difficulty.json`: Multiple full trials concatenated without labels to evaluate adaptive behavior implicitly.
5. `5_long_continuous.json`: A massive multi-trial block to simulate a continuous multi-hour hearing aid session.

## 6. Validation Results
The validation script successfully parsed and generated streams for all 5 scenarios. 
- Overlapping windows accurately span scene boundaries.
- Timestamps are strictly sequential.
- The `ground_truth` variable flips precisely at the specified transition thresholds.

## 7. Engineering Discussion
This generator represents the final bridge between academic neural network research and embedded product engineering. By using JSON to explicitly script auditory "Scenes", we can now guarantee deterministic, reproducible evaluations of the Policy Engine. Instead of using summary statistics over a batch dataset, we can pinpoint exact switch latencies down to the 50ms hop.

## 8. Future DTU Integration Plan
When Domain Shift robustness needs to be evaluated (e.g., Phase 16.3), the integration plan is:
1. Implement `DTUAdapter(DatasetAdapter)` mirroring the `KULAdapter`.
2. Register it with the `ContinuousSessionGenerator`: `adapters={'KUL': KULAdapter(), 'DTU': DTUAdapter()}`.
3. Write a new JSON scenario `domain_shift.json` that defines `{"dataset": "KUL"}` for Scene 1 and `{"dataset": "DTU"}` for Scene 2.
4. The Generator will natively handle the dataset context switch in the middle of the continuous stream.
