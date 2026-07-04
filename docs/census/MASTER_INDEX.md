# MASTER INDEX — EEG AAD Project Scientific Census

> **Generated**: 2026-07-02 from complete repository scan (451 files, 280 Python, 37 Markdown, 15 models, 42 training scripts, 179 analysis scripts)

---

## Document Registry

| # | Document | Purpose |
|---|----------|---------|
| 1 | [MASTER_PROJECT_STATE.md](file:///C:/Users/lokes/.gemini/antigravity-ide/brain/0ad5df83-2069-46f1-8ad2-402fd215df23/MASTER_PROJECT_STATE.md) | Complete canonical repository report — every phase, every result |
| 2 | [MASTER_METRICS_DATABASE.md](file:///C:/Users/lokes/.gemini/antigravity-ide/brain/0ad5df83-2069-46f1-8ad2-402fd215df23/MASTER_METRICS_DATABASE.md) | All numerical results — exact values, no rounding |
| 3 | [MASTER_TIMELINE.md](file:///C:/Users/lokes/.gemini/antigravity-ide/brain/0ad5df83-2069-46f1-8ad2-402fd215df23/MASTER_TIMELINE.md) | Project chronology — phase ordering, motivations, transitions |
| 4 | [MASTER_DISCOVERIES.md](file:///C:/Users/lokes/.gemini/antigravity-ide/brain/0ad5df83-2069-46f1-8ad2-402fd215df23/MASTER_DISCOVERIES.md) | All scientific discoveries and insights |
| 5 | [MASTER_FAILURES.md](file:///C:/Users/lokes/.gemini/antigravity-ide/brain/0ad5df83-2069-46f1-8ad2-402fd215df23/MASTER_FAILURES.md) | All failed hypotheses, bugs, and dead ends |
| 6 | [MASTER_MODELS.md](file:///C:/Users/lokes/.gemini/antigravity-ide/brain/0ad5df83-2069-46f1-8ad2-402fd215df23/MASTER_MODELS.md) | All architectures — purpose, params, results, status |
| 7 | [MASTER_DATASETS.md](file:///C:/Users/lokes/.gemini/antigravity-ide/brain/0ad5df83-2069-46f1-8ad2-402fd215df23/MASTER_DATASETS.md) | All dataset information — DTU, KUL, preprocessing |
| 8 | [MASTER_PRODUCT_STATUS.md](file:///C:/Users/lokes/.gemini/antigravity-ide/brain/0ad5df83-2069-46f1-8ad2-402fd215df23/MASTER_PRODUCT_STATUS.md) | Current hearing-aid platform status |
| 9 | [MASTER_ROADMAP.md](file:///C:/Users/lokes/.gemini/antigravity-ide/brain/0ad5df83-2069-46f1-8ad2-402fd215df23/MASTER_ROADMAP.md) | Future roadmap — immediate through long-term |
| 10 | **MASTER_INDEX.md** (this file) | Navigable index linking every report |

---

## Repository at a Glance

```
Repository: EEG_8_Channel_Pipeline (Loki3306)
Total Files: 451
Python: 280 | Markdown: 37 | MATLAB: 1
Models: 15 | Training Scripts: 42 | Analysis Scripts: 179
Evaluation Scripts: 12 | Decision Engine Components: 3
Scenario Definitions: 5 | Result Directories: 7

Primary Datasets: DTU (18 subjects), KUL (16 subjects)
Primary Model: AAD-Conformer (KUL) / ContrastiveMatchNet (DTU)
Current Phase: 17.3 — Product Metrics Redesign
```

## Quick Reference: Key Results

| Metric | Value | Source |
|--------|-------|--------|
| Conformer LOSO Trial Accuracy (KUL, 5-seed mean) | 77.12% ± 9.99% | conformer_loso_summary.csv |
| Ridge Baseline (DTU, 2ch, 10s) | 55.19% | ridge_loso_summary.json |
| ContrastiveMatchNet Window Accuracy (DTU) | 69.02% | Phase 2 export |
| Confidence AUROC (5-feature XGBoost) | 0.8057 | Paper Foundation |
| Selective Accuracy @ 70% Coverage | 81.55% | Behavior audit |
| Learned Confidence Head AUROC (Conformer) | 0.7337 | Final AAD Report |
| Learned Confidence Head ECE | 0.0998 | Final AAD Report |
| Cross-Dataset Zero-Shot (KUL→DTU, Acc. Pearson) | 68.24% | Phase 10 |
| Cross-Dataset Zero-Shot (KUL→DTU, Majority Vote) | 54.26% | Phase 10 |
| Phase 17.2 Controller: True Switches | 2 | phase17_2_report.md |
| Phase 17.2 Controller: False Switches | 2 | phase17_2_report.md |
| Phase 17.2 Controller: Precision | 50.0% | phase17_2_report.md |
| Phase 17.2 Controller: Coverage | 92.58% | phase17_2_report.md |
| Phase 17.3 UX: Audible False Switches/hr | 22.53 | Phase 17.3 output |
| Phase 17.3 UX: Decision Availability | 99.63% | Phase 17.3 output |
| Phase 17.3 UX: Correct Lock Coverage | 84.48% | Phase 17.3 output |
| Phase 17.3 UX: Acquisition Latency | 4.99s | Phase 17.3 output |
| Phase 17.3 UX: Switch/Recovery Latency | 25.41s | Phase 17.3 output |
