# MCP Tool Reference

This reference documents all MCP tools exposed by the EEG MCP server.

---

## 1. `autonomous_review`
- **Arguments**:
  - `files` (list of strings): Paths to target files.
  - `task` (string, optional): Prompt/goal description.
  - `max_iterations` (integer, default `3`): Max allowed fix loops.
- **Return Type**: `str` (JSON block representing status reports, or the correction instruction prompt)
- **Purpose**: Runs a multi-stage review loop (Git checkpoint, Ruff validation, compile checks, EEG validator, ChatGPT Review, and Independent Verification).
- **Example**:
  ```python
  autonomous_review(files=["scratch/flawed_eeg.py"], task="Validate EEG script", max_iterations=3)
  ```

---

## 2. `search_experiment_memory`
- **Arguments**:
  - `query` (string): Query text to filter lessons database.
- **Return Type**: `str` (Markdown list of matching past experiments, summaries, and lessons)
- **Purpose**: Searches through `.research/index.json` using lightweight keyword lookup.
- **Example**:
  ```python
  search_experiment_memory(query="loso")
  ```

---

## 3. `review_experiment`
- **Arguments**:
  - `files` (list of strings): Target files.
- **Return Type**: `str` (JSON object detailing issues, warnings, and matching memory context)
- **Purpose**: Domain heuristic validation for EEG experiment design errors.
- **Example**:
  ```python
  review_experiment(files=["scratch/flawed_eeg.py"])
  ```

---

## 4. `review_code`
- **Arguments**:
  - `files` (list of strings): Target files.
  - `task` (string, optional): Goal description.
- **Return Type**: `str` (Standard JSON payload from ChatGPT)
- **Purpose**: Routes code and matching research memory context to the browser bridge for ChatGPT review.
- **Example**:
  ```python
  review_code(files=["scratch/flawed_eeg.py"], task="Validate EEG script")
  ```

---

## 5. `create_checkpoint` / `restore_checkpoint` / `discard_checkpoint`
- **Purpose**: Internal helpers exposed as tools to manage temporary git states using stash lists.
