# PROJECT STATE

## Current Phase:
Phase 40.5: Combined Adaptation Study

## Current Best Model:
Neural Ridge Hybrid (Zero-shot KUL Conformer backbone with PROJECTION adaptation)

## Current Best Result:
* Phase 40.5 (PROJECTION Adaptation): **59.28% AUROC**.
* Phase 40.5 (HYBRID Adaptation): 59.15% AUROC.
* Phase 40.5 (LATENT Adaptation): 58.35% AUROC.
* Phase 39 (Zero-Shot Transfer + Neural Ridge, 8 channels, 5s window): **57.48% AUROC**. 
* Falsification tests passed (Shuffle Audio: 51.3%, Noise: 44.3%).
* Best Historical (KUL): 75.8% AUROC (Phase 28 Conformer, 30s window).

## Current Dataset:
AASD (Auditory Attention Switching Dataset).
* Constraint: Hardware restricted to 8 physical channels.

## Current Focus
Our primary goal is to **Train a native Conformer model from scratch on the AASD Dataset**.
Phase 41 rigorously proved that KUL-pretrained Conformer filters do not generalize to the 8-channel AASD topography for the vast majority of subjects. Zero-shot transfer (mean AUROC 0.5028) and lightweight projection adapters (mean AUROC 0.5082) fail to rescue performance dataset-wide. We are abandoning the frozen KUL backbone strategy.

## Recent Milestones
- **Phase 41**: Completed Dataset-Wide Within-Subject Validation across 18 AASD subjects. Discovered that S18 was a statistical anomaly and zero-shot KUL->AASD transfer fails catastrophically for most subjects.
- **Phase 40.5**: Confirmed `PROJECTION_ONLY` adaptation was the best strategy on S18 (59.28% AUROC).
- **Phase 40**: Demonstrated layer-wise adaptation capabilities using the newly stabilized backward pass, achieving 59.06% AUROC on target subject S18 via `backbone.upsample` training.

## Known Invalid Hypotheses:
- **AASD Labels Corrupted:** False. Triggers are mathematically perfectly aligned.
- **Conformer is SOTA on AASD Switch Data:** False. High latent inertia prevents rapid switching without architectural modifications (Ridge adaptation).
- **MatchNet outperforms Regression:** False. MatchNet memorizes the training data heavily.
- **Fine-Tuning Latents is highly effective for cross-dataset transfer:** False. The `ResidualLatentAdapter` barely improved the analytical zero-shot baseline (0.5730 -> 0.5748). Under the evaluated architecture, latent adaptation alone produces negligible gains, suggesting adaptation earlier in the network—including the spatial frontend—may be more promising.

## Open Questions:
- Which layer of the network suffers the most from domain shift during cross-dataset transfer? (Requires Layer-wise adaptation study).
- What is the biological threshold of latency in Auditory Attention Switching?

## Blockers:
- None. Phase 39 completed successfully.
