# EEG Autonomous Engineering Pipeline v1.0

This document describes the high-level architecture, design specifications, and pipeline flow for the EEG Autonomous Engineering Pipeline v1.0.

## 🌟 Overall Architecture
The system consists of an MCP-based orchestrator, a browser bridge interfacing directly with ChatGPT, local validation pipelines (static checks and domain heuristics), independent verifiers, and persistent filesystem databases for experiment audit trails and research memory.

### Key Subsystems:
1. **MCP Server Orchestration**: Manages task iteration loops, tracks file changes, manages Git stashes, and coordinates validation phases.
2. **ChatGPT Browser Bridge**: Uses a persistent Chrome user data profile and an asynchronous IPC script (`antigravity_chatgpt_ipc.py`) to poll for requests, invoke ChatGPT, and return structured payloads.
3. **EEG Research Validator**: Enforces domain constraints such as Leave-One-Subject-Out (LOSO) cross-validation and temporal projection alignments.
4. **Independent Verifier**: Double-checks whether a suggested fix resolved previous issues and ensures no regressions were introduced.
5. **Persistent Audit Trail**: Stores execution logs, patches, and detailed review results for every loop iteration in a reproducible `.runs/` directory.
6. **Research Memory**: Implements keyword-matching retrieval to inject historical lessons learned directly into current code reviews.

---

## 🛠️ Repository Layout
```text
EEG_Training_New/
├── .mcp/
│   └── server.py             # Main MCP Server implementation
├── .research/
│   ├── experiments/          # Saved JSON metadata for runs
│   ├── lessons/              # Lesson repositories
│   └── index.json            # Central memory index
├── .runs/
│   └── <timestamp>/          # Immutable runs history
├── docs/                     # Documentation files
└── scratch/
    ├── antigravity_chatgpt_ipc.py # Browser IPC automation script
    ├── prompt.txt            # Interface files for Playwright IPC
    └── response.txt          # Response files for Playwright IPC
```

---

## 🔄 Execution Flow
```text
Task Triggered
      ↓
Create Git Checkpoint (git stash)
      ↓
Static Validation (Ruff & Compile)
      ↓
EEG Research Heuristic Validation
      ↓
ChatGPT Initial Review (via Browser Bridge)
      ↓
[If Fail] Apply Autocomplete Fix Loop
      ↓
Diff Review & Independent Verification
      ↓
Save Immutably to .runs/ and .research/
      ↓
Accept PASS / Rollback FAIL
```

---

## ⚠️ Current Limitations
- **Browser automation dependency**: Relies on ChatGPT's DOM structures, rendering it susceptible to UI changes.
- **Single session restriction**: Does not support concurrent execution threads.
- **Retrieval granularity**: Uses basic keyword-based matching for memory injection instead of vector search.
