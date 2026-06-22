# Phase 1: Confidence Benchmarking

## Phase Status

- [x] Step 1.1 Export
- [ ] Step 1.2 Margin Confidence
- [ ] Step 2 Reliability Analysis

**Notes:**
- MatchNet audit complete: verified from actual `.mat` files that `wavA` is ALWAYS the attended stream. `ex.label` (1 or 2) represents whether the attended speaker was Male or Female, not whether A or B was attended.
- The 68.5% LOSO baseline is valid and cleared.
- The initial 50.09% export was due to checking prediction against speaker gender. Export script fixed.
