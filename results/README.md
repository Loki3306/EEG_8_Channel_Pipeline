# Results Tracking Infrastructure

This directory contains persistent artifacts and logging mechanisms for all experiments in the AAD research project. This ensures that no finding is ever lost and that future researchers can trace the entire project history.

## Directory Structure

*   **`experiments/`**: Stores raw output, intermediate logs, and specific outputs for individual experiments.
*   **`figures/`**: Stores generated plots, visualizations (t-SNE, UMAP), and graphs. Every figure must have a timestamp or commit hash baked into its metadata or filename.
*   **`statistics/`**: Stores parsed CSVs of results, model outputs, PSD band powers, and distance matrices.
*   **`reports/`**: Stores auto-generated markdown reports outlining hypotheses, findings, and summaries.
*   **`checkpoints/`**: Stores PyTorch model weights (`best.pth` and `final.pth`) for the folds of an experiment.
*   **`metadata/`**: Contains the critical `experiment_registry.csv`.

## Experiment Naming Convention

Every experiment must be logged in `metadata/experiment_registry.csv`.
The experiment ID should follow the format:
`EXP_YYYYMMDD_HHMM_[type]`

Example:
`EXP_20260615_1200_LOSO_Baseline`

Always record the `commit_hash` so that code behavior can be retroactively verified.
