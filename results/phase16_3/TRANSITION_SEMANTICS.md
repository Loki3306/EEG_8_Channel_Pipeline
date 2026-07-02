# Hardware Emulator: Transition Semantics (v1.0)

This document formalizes the exact definition of a "transition" and how switch latency is calculated when evaluating controllers on the Phase 16 Hardware Emulator (`ContinuousSessionGenerator`).

## 1. The Splice Timestamp (Ground Truth)
When a scenario JSON defines a scene sequence:
```json
{
  "scene_name": "Speaker A",
  "duration_seconds": 120.0
},
{
  "scene_name": "Speaker B",
  "duration_seconds": 60.0
}
```
The **Splice Timestamp** is mathematically exactly `120.00` seconds. At sample index `120.00 * 64 = 7680`, the raw biological data instantaneously cuts from the Trial A recording to the Trial B recording.

## 2. The Center-Sample Convention
The emulator slides a 2-second overlapping window over the splice. It does **not** stop at the boundary. 

Metadata for a window is assigned based on the exact **Center Sample** of the window.
- Window `[118.98s, 120.98s]` -> Center `119.98s` -> Labeled as **Speaker A**.
- Window `[119.03s, 121.03s]` -> Center `120.03s` -> Labeled as **Speaker B**.

### Why not ignore transitional windows?
Windows that straddle the boundary contain mixed biological states (e.g. 1.5 seconds of Speaker A, 0.5 seconds of Speaker B). We intentionally pass these to the decoder. This forces the controller to deal with the biological "blur" of a transition natively. If the decoder outputs low confidence during the cross-fade, the `DecisionPolicyEngine` must handle it.

## 3. Official Definition of Switch Latency
When evaluating a controller, Latency is defined relative to the JSON Splice Timestamp.

```text
Latency = Controller_Lock_Timestamp - Splice_Timestamp
```

If the splice occurs at `120.00s`, and the controller transitions its internal state to `LOCKED_B` at `121.50s` (the center timestamp of the window that triggered the state change), the official latency is **+1.50 seconds**.

## 4. Product Metric Definitions
For Phase 17 and beyond, controllers are graded on:
- **Mean Lock Latency**: The average delay between the JSON Splice Timestamp and the controller entering the new `LOCKED` state.
- **False Switch Rate**: Any transition into a `LOCKED` state that does not match the JSON Ground Truth.
- **Coverage**: The percentage of total time the controller was in a correct `LOCKED` state (usable hearing aid output).
- **Time in UNCERTAIN**: Total time the controller spent in the `UNCERTAIN` state.
- **Stable Listening Time**: Average continuous duration spent in a single correct `LOCKED` state without dropping to `UNCERTAIN`.
- **Oscillation Frequency**: The number of state transitions back-and-forth per minute.
