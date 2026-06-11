# PROJECT STATE — EEG AAD (DO NOT IGNORE)

This file defines the CURRENT TRUE STATE of the project.
All future work must align with this.
If anything contradicts this, this file is correct.

---

## CURRENT TASK (HIGHEST PRIORITY)

We have established the ridge baseline and tested small CNN variants.

Current focus:
- improve signal extraction and training objective quality
- preserve the strict LOSO protocol
- keep the 2-channel setup as the default baseline

Implementation goals:
1. Keep the reconstruction pipeline intact.
2. Add train-only target preprocessing with Hilbert envelope extraction, power-law compression, and lowpass filtering.
3. Add configurable lag windows in milliseconds.
4. Add longer evaluation windows (10s, 20s, 30s).
5. Add a contrastive EEG-audio alignment objective as an additional mode.

Do NOT:
- remove reconstruction training
- introduce transformers or large architectures
- change the LOSO evaluation protocol

---

## CURRENT FINDINGS (DO NOT RE-DERIVE)

- EEG and audio are aligned (3200 samples @ 64 Hz)
- wavA and wavB are NOT ordered by label
- Labels represent speaker identity (not A/B index)
- Audio clustering successfully separates 2 speakers (cluster 0 and 1)
- Clustering is consistent across all trials (distinct_cluster_rate = 1.0)

---

## KNOWN ISSUE

Direct correlation-based validation (even with lags) gives ~0.56 accuracy.

This means:
- Model capacity is not the bottleneck
- The useful signal appears weak under strict 2-channel LOSO

---

## EXPECTED INPUT / OUTPUT

Input:
- cluster_of_wavA (per trial)
- cluster_of_wavB (per trial)
- label (1 or 2 per trial)

Output:
- mapping: cluster → label
- validation accuracy after mapping (>0.6 expected)

If accuracy < 0.6:
- mapping is incorrect OR logic is flawed

---

## NEXT STEPS
- [ ] Validate reconstruction objective with new target preprocessing
- [ ] Compare 10s, 20s, and 30s evaluation windows
- [ ] Benchmark contrastive alignment against reconstruction
- [ ] Keep tracking whether the ceiling moves above the current ~56% range
