# 10 Runtime System

## Deployment Architecture
The proposed runtime system for a selective AAD hearing aid operates in a streaming fashion.

```text
[Streaming EEG] + [Streaming Audio A, Audio B]
       |                    |
       v                    v
  [Window Buffer (3s, 1.5s stride)]
                 |
                 v
       [MatchNet Inference]
        - Output: sim_a, sim_b
                 |
                 v
      [Feature Engine State]
        - Compute margin = abs(sim_a - sim_b)
        - Update rolling_std_margin
        - Update trial_consistency
                 |
                 v
        [XGBoost Confidence]
        - Output: Prob(Correct)
                 |
                 v
     [Accept/Reject Decision]
        - Threshold: e.g., > 0.6
                 |
                 +--> If Accept: Output predicted stream (sim_a vs sim_b)
                 +--> If Reject: Maintain previous state / Do nothing
```

## State Tracking
The `Feature Engine State` is a crucial runtime component. It must maintain a lightweight history buffer:
- The last `N` predictions to calculate `trial_consistency`.
- The last `M` margins to calculate `rolling_std_margin`.

This ensures the Confidence model operates with low latency and minimal memory overhead, distinct from the heavy deep learning required by the MatchNet encoder.
