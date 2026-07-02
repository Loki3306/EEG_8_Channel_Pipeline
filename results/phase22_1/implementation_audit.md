# Implementation Audit

- [x] No shared implementation (memory addresses distinct).
- [x] `InfiniteAccumulator` update() and reset() behave correctly.
- [x] `HardCapAccumulator(cap=20.0)` update() and reset() behave correctly.
- [x] `ExponentialDecay(λ=0.9)` update() and reset() behave correctly.
- [x] `AsymmetricDecay(λ=0.5)` update() and reset() behave correctly.
- [x] `SlidingWindow(N=32)` update() and reset() behave correctly.
- [x] `BayesianAccumulator(p_switch=0.01)` update() and reset() behave correctly.
- [x] `CUSUMHybrid(d=0.5, h=3.0)` update() and reset() behave correctly.
- [x] `ShiryaevRobertsHybrid(h=20.0)` update() and reset() behave correctly.
- [x] `PageHinkleyHybrid(h=5.0)` update() and reset() behave correctly.