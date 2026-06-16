import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

def get_git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        raise RuntimeError("Failed to retrieve git commit hash. Refusing to update roadmap without reproducibility metadata.")

def update_roadmap(experiment_name, key_findings, artifacts_produced, next_experiment):
    roadmap_path = REPO_ROOT / "ROADMAP.md"
    
    if not roadmap_path.exists():
        raise FileNotFoundError("ROADMAP.md not found. Cannot append research log.")
        
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    commit = get_git_commit()
    
    log_entry = f"""
## {timestamp}

**Commit:** `{commit}`
**Experiment:** {experiment_name}

### Key Findings
{key_findings}

### Artifacts Produced
{artifacts_produced}

### Next Recommended Experiment
{next_experiment}
"""
    
    with open(roadmap_path, 'a') as f:
        f.write(log_entry)
        
    print(f"Successfully appended research log to ROADMAP.md for experiment '{experiment_name}'.")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python update_roadmap.py <experiment_name> <key_findings> <artifacts_produced> <next_experiment>")
        sys.exit(1)
        
    update_roadmap(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
