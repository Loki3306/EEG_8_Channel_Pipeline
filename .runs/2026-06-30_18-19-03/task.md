Run 3.1 — Frequency Pipeline Verification + Multi-Subject Interpretability Smoke Test. 

Phase 1: Audit frequency ablation pipeline, verify mathematical correctness of frequency filtering. Generate frequency_pipeline_audit.csv and plots.
Phase 2: Select 3 subjects (Strong, Average, Weak) from benchmark. Run Progressive Channel Ablation, Frequency Ablation, Temporal Occlusion, Gradient Saliency on them. Generate CSVs and plots.
Phase 3: Cross-Subject Analysis. Answer if important channels/freqs/saliency remain consistent.
Deliverables: Executive Summary, Verification Results, Files Modified, Multi-Subject Results, Cross-Subject Interpretation, Remaining Risks, Recommendation.
Ensure the orchestrator follows the full confidence-based review pipeline (Developer -> Static -> Anti -> GPT -> Anti Re-review).