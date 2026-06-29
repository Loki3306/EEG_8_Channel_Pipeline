# Repository Map

The structure of the repository, including key directories, is mapped below:

## 🗂️ File Layout

- **`.mcp/`**: Contains the MCP server implementation.
  - [server.py](file:///c:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/.mcp/server.py): Defines MCP tools, validation states, research memory recording, and loop orchestration.
- **`.research/`**: The persistent research database.
  - [index.json](file:///c:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/.research/index.json): Main index containing deduplicated lessons and run metadata.
  - `experiments/`: Timestamped JSON files containing individual experiment summaries and lessons.
- **`.runs/`**: The immutable, reproducible run tracking directory.
  - Entries are created in folders named `YYYY-MM-DD_HH-MM-SS/` containing `execution.log`, `task.md`, intermediate JSON files, patches, and `final_report.json`.
- **`docs/`**: Holds version documentation, architecture maps, and capabilities specifications.
- **`scratch/`**: Location for the Playwright bridge and interface text files.
  - [antigravity_chatgpt_ipc.py](file:///c:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/scratch/antigravity_chatgpt_ipc.py): Playwright script automating user sessions with ChatGPT.
  - `prompt.txt` / `response.txt`: Communication bridges between python execution and playwright processes.
