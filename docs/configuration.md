# Configuration Specification

Below are the configurations and defaults currently frozen in the system:

## ⚙️ Parameters

| Parameter | Configuration Value / Method | File / Path |
| :--- | :--- | :--- |
| **Iteration Limits** | `max_iterations = 3` | `.mcp/server.py:autonomous_review()` |
| **Browser Timeout** | `300 seconds` (1-second polling intervals) | `.mcp/server.py:BrowserChatGPTReviewer.review()` |
| **Playwright User Profile** | `%LOCALAPPDATA%\Google\Chrome\Playwright_ChatGPT` | `scratch/antigravity_chatgpt_ipc.py` |
| **Scratch Folder** | `scratch/` (relative to workspace root) | Workspace root |
| **Research Folder** | `.research/` | Workspace root |
| **Audit Run Folder** | `.runs/` | Workspace root |
| **Checkpoint Strategy** | `git stash push -u` & `git stash apply` | `.mcp/server.py:create_checkpoint()` |
| **Static Validation** | `ruff check --fix` and `python -m py_compile` | `.mcp/server.py:autonomous_review()` |
| **EEG Validation** | Domain rules matching `loso`, `train_test_split`, `sim = einsum`, `mean(dim=1)` | `.mcp/server.py:review_experiment()` |
