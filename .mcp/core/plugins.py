import os
from pathlib import Path

class BasePlugin:
    def name(self) -> str:
        return "Generic"
    
    def detect(self, repo_root: Path) -> bool:
        return True
    
    def load_memory(self, repo_root: Path) -> dict:
        return {}

class EEGPlugin(BasePlugin):
    def name(self) -> str:
        return "EEG"
    
    def detect(self, repo_root: Path) -> bool:
        # Check for files/dirs with EEG, DTU, KUL, LOSO, MatchNet
        for root, dirs, files in os.walk(repo_root):
            # Prune ignore dirs
            dirs[:] = [d for d in dirs if d not in {".git", ".venv", "venv", "__pycache__", "node_modules"}]
            if any(x in root.lower() or x in "".join(dirs).lower() or x in "".join(files).lower() 
                   for x in ["eeg", "loso", "matchnet", "kul", "dtu"]):
                return True
        return False
        
    def load_memory(self, repo_root: Path) -> dict:
        # Load index.json from .research/
        import json
        research_dir = repo_root / ".research"
        index_file = research_dir / "index.json"
        if index_file.exists():
            try:
                with open(index_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return {"previous_runs": data}
            except Exception:
                pass
        return {}

class ReactPlugin(BasePlugin):
    def name(self) -> str:
        return "React"
        
    def detect(self, repo_root: Path) -> bool:
        # Check for package.json containing react or react-dom, or directories like node_modules
        package_json = repo_root / "package.json"
        if package_json.exists():
            try:
                with open(package_json, "r", encoding="utf-8") as f:
                    content = f.read()
                    if "react" in content.lower():
                        return True
            except Exception:
                pass
        return False

PLUGINS = [EEGPlugin(), ReactPlugin(), BasePlugin()]

def detect_project_plugin(repo_root: Path) -> BasePlugin:
    for plugin in PLUGINS[:-1]:
        if plugin.detect(repo_root):
            return plugin
    return PLUGINS[-1]
