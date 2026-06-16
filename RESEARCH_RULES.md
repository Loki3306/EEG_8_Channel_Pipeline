# Research Integrity Rules

This file governs all analysis, reporting, and experimentation within the repository.

## Rule 1: Never Generate Fake Data
Forbidden:
* Random accuracies
* Mock LOSO scores
* Placeholder statistics
* Synthetic subject metrics
* Random baselines

If required inputs are missing, scripts MUST `raise FileNotFoundError` or `ValueError`. Never silently mock data to continue execution.

## Rule 2: Fail Loudly
If any required file is missing (accuracy CSV, model outputs, checkpoints, subject metadata), the script must terminate execution immediately. Do not substitute defaults.

## Rule 3: Separate Exploratory vs Confirmed Results
Every report must indicate its status:
* **Exploratory**
* **Confirmed**

Confirmed requires real data, a reproducible script, saved metadata, and a commit hash.

## Rule 4: No Placeholder Findings
Forbidden phrases:
* "Hypothesis 1: ..."
* "Finding: TBD"
* "Auto-generated insight"

Reports must only contain findings derived from computed statistics.

## Rule 5: Reproducibility Metadata Mandatory
Every generated artifact (CSV, report, figure) must include:
* UTC timestamp
* git commit hash
* dataset name
* script name
* configuration file details

## Rule 6: Correlation Safety
Any correlation analysis must report:
* Pearson r
* p-value
* sample size
* multiple-comparison correction

No naked correlations.

## Rule 7: Statistical Significance Required
Comparisons between groups must compute and report:
* t-test or Mann-Whitney U
* Effect size (e.g., Cohen's d or rank-biserial correlation)

## Rule 8: Roadmap Integrity
`ROADMAP.md` acts as an experimental log. Every entry must include Date, Commit, Experiment, Result, Interpretation, and Next Step. No vague notes.
