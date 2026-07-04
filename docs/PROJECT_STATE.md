# PROJECT STATE

## Current Phase:
Phase 39: Transfer Learning to AASD

## Current Best Model:
Neural Ridge Hybrid (Frozen KUL Conformer Backbone + Trainable Ridge Decoder + Residual Latent Adapter)

## Current Best Result:
* Phase 39 (Zero-Shot Transfer + Neural Ridge, 8 channels, 5s window): **57.48% AUROC**. 
* Falsification tests passed (Shuffle Audio: 51.3%, Noise: 44.3%).
* Best Historical (KUL): 75.8% AUROC (Phase 28 Conformer, 30s window).

## Current Dataset:
AASD (Auditory Attention Switching Dataset).
* Constraint: Hardware restricted to 8 physical channels.

## Known Invalid Hypotheses:
- **AASD Labels Corrupted:** False. Triggers are mathematically perfectly aligned.
- **Conformer is SOTA on AASD Switch Data:** False. High latent inertia prevents rapid switching without architectural modifications (Ridge adaptation).
- **MatchNet outperforms Regression:** False. MatchNet memorizes the training data heavily.
- **Fine-Tuning Latents is highly effective for cross-dataset transfer:** False. The `ResidualLatentAdapter` barely improved the analytical zero-shot baseline (0.5730 -> 0.5748), implying that the spatial filters (which were frozen) are the true bottleneck across hardware caps.

## Open Questions:
- Since zero-shot cross-dataset transfer hits a ceiling around 57.5% due to hardware differences, do we need to train a cross-dataset spatial adapter (rather than latent adapter)?
- What is the biological threshold of latency in Auditory Attention Switching?

## Blockers:
- None. Phase 39 completed successfully.
