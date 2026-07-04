# PROJECT STATE

## Current Phase:
Phase 39: Transfer Learning to AASD

## Current Best Model:
Neural Ridge Hybrid (Frozen KUL Conformer Backbone + Trainable Ridge Decoder + Residual Latent Adapter)

## Current Best Result:
* Pending Kaggle Execution for Phase 39 Zero-Shot Transfer.
* Best Historical (KUL): 75.8% AUROC (Phase 28 Conformer, 30s window).

## Current Dataset:
AASD (Auditory Attention Switching Dataset).
* Constraint: Hardware restricted to 8 physical channels.

## Known Invalid Hypotheses:
- **AASD Labels Corrupted:** False. Triggers are mathematically perfectly aligned.
- **Conformer is SOTA on AASD Switch Data:** False. High latent inertia prevents rapid switching without architectural modifications (Ridge adaptation).
- **MatchNet outperforms Regression:** False. MatchNet memorizes the training data heavily.

## Open Questions:
- Will the Zero-Shot Ridge weights successfully transfer to AASD using the mapped 8 physical channels?
- Can the `ResidualLatentAdapter` fine-tune enough to adjust for skull thickness variations across users without overfitting?
- What is the biological threshold of latency in Auditory Attention Switching?

## Blockers:
- Waiting on Kaggle execution for Phase 39.
