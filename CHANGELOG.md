# Changelog

All notable changes to this project will be documented in this file.

---

## [1.0.0] - 2026-06-29

### Added
- **MCP Server Framework**: Constructed a custom MCP server (`server.py`) supporting git stashes, compilation checks, and review operations.
- **ChatGPT Browser Bridge**: Built `antigravity_chatgpt_ipc.py` to automate reviews through Playwright chromium instances.
- **EEG Heuristics Validator**: Added code auditing checks for LOSO cross-validation leakage and InfoNCE temporal shape matching.
- **Git Safety Subsystem**: Added automatic stashes (`create_checkpoint`, `restore_checkpoint`) before and after editing files.
- **Diff-Based Review**: Optimized evaluations on subsequent loop iterations by reviewing diffs instead of whole files.
- **Independent Verification stage**: Introduced `verify_fix()` tool to double-check fixes.
- **Persistent Experiment Audit Trail**: Generated timestamped folders under `.runs/` containing logs, patches, and reports.
- **Persistent Research Memory**: Implemented keyword-indexed lessons database under `.research/` to retrieve lessons automatically.
