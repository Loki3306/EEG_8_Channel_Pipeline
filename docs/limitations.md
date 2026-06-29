# System Limitations

Below are the known limitations of the current EEG Autonomous Engineering Pipeline v1.0.

---

## ⚠️ Known Limitations

- **Browser UI Automation Dependency**: The pipeline communicates with ChatGPT via DOM selectors. If OpenAI modifies the ChatGPT interface layouts, the bridge script may require updates.
- **Blocked Active Session**: The Playwright Chromium process locks the persistent Chrome user profile directory. As a result, Chrome cannot be run concurrently outside of the script context.
- **Single Process Queue**: Only one autonomous review thread can execute at a time. Concurrency is not supported due to the single shared profile lock.
- **Keyword-based Retrieval**: The persistent research memory uses simple keyword and word-splitting queries to retrieve relevant experiments, rather than dense semantic vector retrieval.
- **Local Compiler Context**: PyCompile checks run against the local interpreter path and may fail to resolve virtual environment dependencies if packages are missing.
