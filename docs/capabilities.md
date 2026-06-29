# System Capabilities Matrix

The following capabilities are implemented and fully operational in Version 1.0.

---

## 🛠️ Feature Matrix

- **Repository Intelligence**: Gathers context on files, imports, and related symbols to supply ChatGPT with high-quality local prompts.
- **Browser Bridge**: Automates conversational flows with ChatGPT via Playwright Chromium using a persistent browser profile.
- **Autonomous Review**: Coordinates the iterative loop (Static checking, EEG heuristics, ChatGPT prompts, and Independent Verification).
- **Independent Verification**: Invokes a dedicated verifier stage that prevents new bugs from slipping through.
- **Persistent Audit Trail**: Stores timing metrics, diffs, logs, and JSON payloads in `.runs/`.
- **Research Memory**: Automatically writes lessons to `.research/index.json` and injects them into subsequent prompt headers.
- **Static Validation**: Automatically executes `ruff check --fix` and `py_compile` compiler checks.
- **Git Safety**: Creates automatic stashes to guarantee rollback capability if loops fail.
- **EEG Heuristics Validation**: Prevents methodological flaws like LOSO subject leakage and temporal collapse before contrastive learning.
- **Diff Review**: Reviews git diffs instead of complete files on subsequent iterations to optimize token usage.
