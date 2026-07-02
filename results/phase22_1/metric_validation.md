# Metric Validation

Based on the static analysis of `analysis/phase22_benchmark.py`:

- **Correct Coverage**: Computed as `len(correct) / len(trace_df)`. This is strictly **window-based**. It measures the percentage of frames where the `active_lock` matches the `ground_truth`.
- **Wrong Coverage**: Computed similarly on a per-window basis where `active_lock` does not match `ground_truth` and is not `None`.
- **Availability**: Percentage of windows where `active_lock` is not `None`.
- **Oscillations**: Logged natively by the `DecisionPolicyEngine` whenever a switch occurs. This matches the standard implementation.
- **Switch Latency**: Measures the duration from a ground_truth splice until the first window where `active_lock` matches the new ground truth.

**Conclusion**: The metric implementation is correct and matches the intended window-based definitions. The ~51% coverage is NOT an evaluation artifact; it literally means the finite-memory strategies spend half of their time holding the wrong lock.
