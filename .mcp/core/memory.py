import json
from pathlib import Path
from typing import List, Dict

def load_index_memory(repo_root: Path) -> List[Dict]:
    index_file = repo_root / ".research" / "index.json"
    if index_file.exists():
        try:
            with open(index_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []

def search_memory(repo_root: Path, query: str) -> Dict:
    index_data = load_index_memory(repo_root)
    query_lower = query.lower()
    matches = []
    
    for entry in index_data:
        searchable = f"{entry.get('task', '')} {entry.get('summary', '')} {' '.join(entry.get('keywords', []))}".lower()
        if query_lower in searchable or any(word in searchable for word in query_lower.split() if len(word) > 3):
            matches.append(entry)
            
    # Collect lessons, summaries, decisions
    lessons = []
    decisions = []
    experiments = []
    
    for m in matches:
        experiments.append({
            "experiment_id": m.get("experiment_id"),
            "task": m.get("task"),
            "status": m.get("status"),
            "summary": m.get("summary")
        })
        lessons.extend(m.get("lessons", []))
        if "decision" in m:
            decisions.append(m["decision"])
            
    # Deduplicate lessons
    deduped_lessons = list(dict.fromkeys(lessons))
    
    return {
        "experiments": experiments,
        "lessons": deduped_lessons,
        "decisions": decisions
    }
