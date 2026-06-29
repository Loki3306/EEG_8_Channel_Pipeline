import os
import subprocess
from pathlib import Path
from typing import List

from mcp.server.fastmcp import FastMCP

# Initialize the MCP server
mcp = FastMCP("EEG MCP")

# --- Helper Functions ---
import abc

class LLMReviewer(abc.ABC):
    @abc.abstractmethod
    def review(self, prompt: str) -> str:
        pass

class MockReviewer(LLMReviewer):
    def review(self, prompt: str) -> str:
        return (
            "Overall Verdict\n\nPASS\n\n"
            "Confidence\n\n85\n\n"
            "Blocking Issues\n\n- None detected in mock mode.\n\n"
            "Warnings\n\n- Ensure dataset loading handles missing files gracefully.\n\n"
            "Suggestions\n\n- Consider adding type hints to all function signatures.\n\n"
            "Files Reviewed\n\n- (Mocked file list)\n\n"
            "Repository Context Used\n\n- (Mocked context gathered from repo)\n"
        )

def get_reviewer() -> LLMReviewer:
    from reviewers.browser_chatgpt import BrowserChatGPTReviewer
    return BrowserChatGPTReviewer()

def get_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent

def get_ignore_dirs() -> set:
    return {
        ".git", ".venv", "venv", "__pycache__", ".pytest_cache", 
        "node_modules", ".idea", ".vscode"
    }

def is_binary(file_path: Path) -> bool:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            f.read(1024)
        return False
    except UnicodeDecodeError:
        return True

import ast

def _get_imports(file_path: Path) -> set:
    imports = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=str(file_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
                    for alias in node.names:
                        imports.add(f"{node.module}.{alias.name}")
    except Exception:
        pass
    return imports

def _get_classes(file_path: Path) -> list:
    classes = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=str(file_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
    except Exception:
        pass
    return classes

# --- Existing Tools ---
@mcp.tool()
def list_project_files() -> str:
    """Returns a newline-separated list of relative file paths in the repository root."""
    repo_root = get_repo_root()
    ignore_dirs = get_ignore_dirs()
    
    file_paths = []
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith('.')]
        for file in files:
            full_path = Path(root) / file
            rel_path = full_path.relative_to(repo_root).as_posix()
            file_paths.append(rel_path)
            
    return "\n".join(file_paths)

@mcp.tool()
def search_project(query: str) -> str:
    """Search the repository for filenames matching the query."""
    repo_root = get_repo_root()
    ignore_dirs = get_ignore_dirs()
    
    query_lower = query.lower()
    matching_files = []
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith('.')]
        for file in files:
            if query_lower in file.lower():
                full_path = Path(root) / file
                rel_path = full_path.relative_to(repo_root).as_posix()
                matching_files.append(rel_path)
                
    if not matching_files:
        return "No matching files found."
    return "\n".join(matching_files)

# --- New Tools ---

@mcp.tool()
def read_file(path: str) -> str:
    """Read a UTF-8 text file from the repository."""
    repo_root = get_repo_root()
    target_path = (repo_root / path).resolve()
    
    if not target_path.is_relative_to(repo_root):
        return "Error: Path traversal detected. Access denied."
        
    if not target_path.exists():
        return f"Error: File '{path}' does not exist."
        
    if not target_path.is_file():
        return f"Error: '{path}' is not a file."
        
    if is_binary(target_path):
        return "Binary file not supported."
        
    try:
        with open(target_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

@mcp.tool()
def read_files(paths: list[str]) -> str:
    """Read multiple UTF-8 text files from the repository."""
    results = []
    for path in paths:
        results.append(f"========== {path} ==========\n{read_file(path)}")
    return "\n\n".join(results)

@mcp.tool()
def grep(pattern: str) -> str:
    """Search every repository text file for a pattern (case-insensitive)."""
    repo_root = get_repo_root()
    ignore_dirs = get_ignore_dirs()
    pattern_lower = pattern.lower()
    results = []
    
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith('.')]
        for file in files:
            full_path = Path(root) / file
            if is_binary(full_path):
                continue
                
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                rel_path = full_path.relative_to(repo_root).as_posix()
                for i, line in enumerate(lines, 1):
                    if pattern_lower in line.lower():
                        results.append(f"{rel_path}:{i}:{line.rstrip()}")
            except (UnicodeDecodeError, Exception):
                pass
                
    if not results:
        return "No matches found."
    return "\n".join(results)

@mcp.tool()
def get_git_status() -> str:
    """Return the current git status."""
    repo_root = get_repo_root()
    try:
        result = subprocess.run(["git", "status"], cwd=repo_root, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error running git status: {e.stderr}"
    except Exception as e:
        return f"Error: {e}"

@mcp.tool()
def get_git_diff() -> str:
    """Return the current git diff."""
    repo_root = get_repo_root()
    try:
        result = subprocess.run(["git", "diff"], cwd=repo_root, capture_output=True, text=True, check=True)
        if not result.stdout.strip():
            return "Working tree clean."
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error running git diff: {e.stderr}"
    except Exception as e:
        return f"Error: {e}"

@mcp.tool()
def get_file_tree() -> str:
    """Return a compact directory tree up to depth 3."""
    repo_root = get_repo_root()
    ignore_dirs = get_ignore_dirs()
    max_depth = 3
    
    tree_lines = []
    
    def walk_tree(current_path: Path, current_depth: int):
        if current_depth > max_depth:
            return
            
        try:
            items = sorted(current_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
            items = [item for item in items if item.name not in ignore_dirs and not item.name.startswith('.')]
            
            indent = "  " * (current_depth - 1)
            for item in items:
                if item.is_dir():
                    tree_lines.append(f"{indent}{item.name}/")
                    walk_tree(item, current_depth + 1)
                else:
                    tree_lines.append(f"{indent}{item.name}")
        except PermissionError:
            pass
            
    walk_tree(repo_root, 1)
    return "\n".join(tree_lines)

@mcp.tool()
def find_training_pipeline(model_name: str) -> str:
    """Find training scripts for a specific model."""
    repo_root = get_repo_root()
    ignore_dirs = get_ignore_dirs()
    target_module = f"models.{model_name}"
    pipelines = []
    
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith('.')]
        for file in files:
            if not file.endswith('.py'):
                continue
            full_path = Path(root) / file
            imports = _get_imports(full_path)
            
            if target_module in imports or any(imp.startswith(target_module) for imp in imports):
                rel_path = full_path.relative_to(repo_root).as_posix()
                pipelines.append(rel_path)
                
    if not pipelines:
        return "No training pipelines found."
    return "\n".join(pipelines)

@mcp.tool()
def find_model_definition(model_name: str) -> str:
    """Find a model's definition, classes, and where it is imported."""
    repo_root = get_repo_root()
    model_path = repo_root / "models" / f"{model_name}.py"
    
    if not model_path.exists():
        ignore_dirs = get_ignore_dirs()
        found = False
        for root, dirs, files in os.walk(repo_root):
            dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith('.')]
            if f"{model_name}.py" in files:
                model_path = Path(root) / f"{model_name}.py"
                found = True
                break
        if not found:
            return f"Model '{model_name}' not found."
            
    rel_model_path = model_path.relative_to(repo_root).as_posix()
    classes = _get_classes(model_path)
    target_module = rel_model_path.replace('/', '.').replace('.py', '')
    
    import_locations = []
    ignore_dirs = get_ignore_dirs()
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith('.')]
        for file in files:
            if not file.endswith('.py'):
                continue
            full_path = Path(root) / file
            if full_path == model_path:
                continue
            imports = _get_imports(full_path)
            if target_module in imports or any(imp.startswith(target_module) for imp in imports):
                import_locations.append(full_path.relative_to(repo_root).as_posix())
                
    result = []
    result.append(f"Model File: {rel_model_path}")
    result.append(f"Classes: {', '.join(classes) if classes else 'None'}")
    result.append(f"Import Locations:")
    if import_locations:
        for loc in import_locations:
            result.append(f"  - {loc}")
    else:
        result.append("  - None")
        
    return "\n".join(result)

@mcp.tool()
def find_dataset(dataset_name: str) -> str:
    """Find all files related to a specific dataset."""
    repo_root = get_repo_root()
    ignore_dirs = get_ignore_dirs()
    dataset_lower = dataset_name.lower()
    
    categories = {
        "Data / Preprocessing": [],
        "Models": [],
        "Training": [],
        "Analysis": [],
        "Documentation": [],
        "Other": []
    }
    
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith('.')]
        rel_dir = Path(root).relative_to(repo_root).as_posix()
        
        for file in files:
            if dataset_lower in file.lower() or dataset_lower in rel_dir.lower():
                full_path = Path(root) / file
                rel_path = full_path.relative_to(repo_root).as_posix()
                
                if rel_dir.startswith("data") or rel_dir.startswith("preprocessing"):
                    categories["Data / Preprocessing"].append(rel_path)
                elif rel_dir.startswith("models"):
                    categories["Models"].append(rel_path)
                elif rel_dir.startswith("training"):
                    categories["Training"].append(rel_path)
                elif rel_dir.startswith("analysis"):
                    categories["Analysis"].append(rel_path)
                elif rel_dir.startswith("docs"):
                    categories["Documentation"].append(rel_path)
                else:
                    categories["Other"].append(rel_path)
                    
    result = []
    for cat, items in categories.items():
        if items:
            result.append(f"=== {cat} ===")
            for item in items:
                result.append(item)
            result.append("")
            
    if not result:
        return f"No files found for dataset '{dataset_name}'."
    return "\n".join(result).strip()

@mcp.tool()
def project_summary() -> str:
    """Automatically inspect the repository and return a structured summary."""
    repo_root = get_repo_root()
    ignore_dirs = get_ignore_dirs()
    
    summary = {
        "Datasets (data/)": [],
        "Models (models/)": [],
        "Training Scripts (training/)": [],
        "Preprocessing (preprocessing/)": [],
        "Analysis (analysis/)": [],
        "Documentation (docs/)": []
    }
    
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith('.')]
        rel_dir = Path(root).relative_to(repo_root).as_posix()
        
        for file in files:
            full_path = Path(root) / file
            rel_path = full_path.relative_to(repo_root).as_posix()
            
            if rel_dir == "data" or rel_dir.startswith("data/"):
                summary["Datasets (data/)"].append(rel_path)
            elif rel_dir == "models" or rel_dir.startswith("models/"):
                summary["Models (models/)"].append(rel_path)
            elif rel_dir == "training" or rel_dir.startswith("training/"):
                summary["Training Scripts (training/)"].append(rel_path)
            elif rel_dir == "preprocessing" or rel_dir.startswith("preprocessing/"):
                summary["Preprocessing (preprocessing/)"].append(rel_path)
            elif rel_dir == "analysis" or rel_dir.startswith("analysis/"):
                summary["Analysis (analysis/)"].append(rel_path)
            elif rel_dir == "docs" or rel_dir.startswith("docs/") or file.endswith(".md"):
                summary["Documentation (docs/)"].append(rel_path)
                
    result = []
    for cat, items in summary.items():
        result.append(f"## {cat} ({len(items)} files)")
        for item in items:
            result.append(f"- {item}")
        result.append("")
        
    return "\n".join(result).strip()

@mcp.tool()
def find_related_files(path: str) -> str:
    """Find related files by imports, imported-by, same module, and similar filename."""
    repo_root = get_repo_root()
    target_path = (repo_root / path).resolve()
    
    if not target_path.exists():
        return f"Error: File '{path}' does not exist."
        
    rel_target = target_path.relative_to(repo_root).as_posix()
    target_module = rel_target.replace('/', '.').replace('.py', '')
    
    imports = _get_imports(target_path)
    local_imports = []
    for imp in imports:
        parts = imp.split('.')
        for i in range(len(parts), 0, -1):
            possible_file = repo_root / ("/".join(parts[:i]) + ".py")
            if possible_file.exists():
                local_imports.append(possible_file.relative_to(repo_root).as_posix())
                break
                
    local_imports = list(set(local_imports))
    
    imported_by = []
    ignore_dirs = get_ignore_dirs()
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith('.')]
        for file in files:
            if not file.endswith('.py'):
                continue
            full_path = Path(root) / file
            if full_path == target_path:
                continue
            file_imports = _get_imports(full_path)
            if target_module in file_imports or any(imp.startswith(target_module) for imp in file_imports):
                imported_by.append(full_path.relative_to(repo_root).as_posix())
                
    same_module = []
    target_dir = target_path.parent
    if target_dir.exists():
        for file in target_dir.iterdir():
            if file.is_file() and file != target_path and file.name not in ignore_dirs and not file.name.startswith('.'):
                same_module.append(file.relative_to(repo_root).as_posix())
                
    similar_filename = []
    target_stem = target_path.stem
    target_words = set(target_stem.split('_'))
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith('.')]
        for file in files:
            full_path = Path(root) / file
            if full_path == target_path:
                continue
            file_stem = full_path.stem
            file_words = set(file_stem.split('_'))
            if len(target_words.intersection(file_words)) >= 2 or target_stem in file_stem or file_stem in target_stem:
                if len(file_stem) > 3 and len(target_stem) > 3:
                    similar_filename.append(full_path.relative_to(repo_root).as_posix())
                    
    similar_filename = list(set(similar_filename) - set(imported_by) - set(local_imports) - set(same_module))
    
    result = []
    result.append(f"=== Related to {path} ===")
    
    result.append("\nImports:")
    for item in sorted(local_imports) or ["None"]:
        result.append(f"  - {item}")
        
    result.append("\nImported-by:")
    for item in sorted(imported_by) or ["None"]:
        result.append(f"  - {item}")
        
    result.append("\nSame directory:")
    for item in sorted(same_module) or ["None"]:
        result.append(f"  - {item}")
        
    result.append("\nSimilar filename:")
    for item in sorted(similar_filename) or ["None"]:
        result.append(f"  - {item}")
        
    return "\n".join(result)

@mcp.tool()
def repository_statistics() -> str:
    """Return repository statistics."""
    repo_root = get_repo_root()
    ignore_dirs = get_ignore_dirs()
    
    stats = {
        "total_files": 0,
        "python_files": 0,
        "markdown_files": 0,
        "matlab_files": 0,
        "models": 0,
        "training_scripts": 0,
        "analysis_scripts": 0
    }
    
    all_files = []
    
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith('.')]
        rel_dir = Path(root).relative_to(repo_root).as_posix()
        
        for file in files:
            stats["total_files"] += 1
            full_path = Path(root) / file
            
            if file.endswith('.py'):
                stats["python_files"] += 1
            elif file.endswith('.md'):
                stats["markdown_files"] += 1
            elif file.endswith('.m'):
                stats["matlab_files"] += 1
                
            if rel_dir == "models" or rel_dir.startswith("models/"):
                stats["models"] += 1
            elif rel_dir == "training" or rel_dir.startswith("training/"):
                stats["training_scripts"] += 1
            elif rel_dir == "analysis" or rel_dir.startswith("analysis/"):
                stats["analysis_scripts"] += 1
                
            try:
                size = full_path.stat().st_size
                all_files.append((size, full_path.relative_to(repo_root).as_posix()))
            except Exception:
                pass
                
    largest_files = sorted(all_files, reverse=True)[:5]
    
    result = [
        f"Total Files: {stats['total_files']}",
        f"Python Files: {stats['python_files']}",
        f"Markdown Files: {stats['markdown_files']}",
        f"MATLAB Files: {stats['matlab_files']}",
        f"Model Count (models/): {stats['models']}",
        f"Training Script Count (training/): {stats['training_scripts']}",
        f"Analysis Script Count (analysis/): {stats['analysis_scripts']}",
        "\nLargest Files:"
    ]
    
    for size, path in largest_files:
        result.append(f"  - {path} ({size / 1024:.2f} KB)")
        
    return "\n".join(result)

@mcp.tool()
def review_code(files: list[str], task: str = "") -> str:
    """Perform an AI-assisted code review using repository context."""
    repo_root = get_repo_root()
    
    # 1. Read files and build context
    file_contents = []
    related_context = []
    
    for f_path in files:
        # Call the existing tools directly
        content = read_file(f_path)
        file_contents.append(f"### {f_path} ###\n{content}\n")
        
        rel_info = find_related_files(f_path)
        related_context.append(f"### Relations for {f_path} ###\n{rel_info}\n")
        
    context_str = "\n".join(related_context)
    content_str = "\n".join(file_contents)
    
    memory_context = search_experiment_memory(task)
    
    # 2. Build Prompt
    prompt = f"""You are an expert AI code reviewer for an EEG research repository.
Task: {task}

{memory_context}

Analyze the following files for:
- Logic errors
- Runtime bugs
- Architecture issues
- Code quality
- Missing edge cases
- Research methodology violations
- Possible train/test leakage
- Incorrect assumptions
- Potential hallucinations
- Suggested improvements

FILE CONTENTS:
{content_str}

REPOSITORY CONTEXT:
{context_str}

Return your review in the following strict JSON structure.
No Markdown.
No prose.
Return ONLY valid JSON.

If the review passes:
{{
  "verdict": "PASS",
  "confidence": 93,
  "blocking_issues": [],
  "warnings": [],
  "suggested_fixes": [],
  "summary": ""
}}

If the review fails:
{{
  "verdict": "FAIL",
  "confidence": 88,
  "blocking_issues": [
    "...",
    "..."
  ],
  "warnings": [
    "..."
  ],
  "suggested_fixes": [
    "..."
  ],
  "summary": ""
}}
"""

    # 3. Send to LLM
    reviewer = get_reviewer()
    result = reviewer.review(prompt)
    
    return result

import json
import time
import subprocess

_REVIEW_STATES = {}

@mcp.tool()
def create_checkpoint(branch_suffix: str) -> str:
    """Create a git checkpoint by stashing."""
    repo_root = get_repo_root()
    msg = f"AUTO_CHECKPOINT_{branch_suffix}"
    try:
        subprocess.run(["git", "stash", "push", "-u", "-m", msg], cwd=repo_root, capture_output=True, text=True, check=True)
        subprocess.run(["git", "stash", "apply"], cwd=repo_root, capture_output=True, check=True)
        return msg
    except Exception as e:
        return f"Error: {e}"

@mcp.tool()
def restore_checkpoint(checkpoint_msg: str) -> str:
    """Restore a git checkpoint."""
    repo_root = get_repo_root()
    try:
        res = subprocess.run(["git", "stash", "list"], cwd=repo_root, capture_output=True, text=True, check=True)
        for line in res.stdout.split('\n'):
            if checkpoint_msg in line:
                stash_idx = line.split(':')[0]
                subprocess.run(["git", "reset", "--hard"], cwd=repo_root, capture_output=True, check=True)
                subprocess.run(["git", "clean", "-fd"], cwd=repo_root, capture_output=True, check=True)
                subprocess.run(["git", "stash", "apply", stash_idx], cwd=repo_root, capture_output=True, check=True)
                return f"Restored {checkpoint_msg}"
        return "Checkpoint not found."
    except Exception as e:
        return str(e)

@mcp.tool()
def discard_checkpoint(checkpoint_msg: str) -> str:
    """Discard a git checkpoint."""
    repo_root = get_repo_root()
    try:
        res = subprocess.run(["git", "stash", "list"], cwd=repo_root, capture_output=True, text=True, check=True)
        for line in res.stdout.split('\n'):
            if checkpoint_msg in line:
                stash_idx = line.split(':')[0]
                subprocess.run(["git", "stash", "drop", stash_idx], cwd=repo_root, capture_output=True, check=True)
                return f"Discarded {checkpoint_msg}"
        return "Checkpoint not found."
    except Exception as e:
        return str(e)

@mcp.tool()
def review_experiment(files: list[str]) -> str:
    """EEG Research Validator."""
    repo_root = get_repo_root()
    issues = []
    warnings = []
    for f in files:
        target = repo_root / f
        if not target.exists():
            continue
        try:
            with open(target, 'r', encoding='utf-8') as file:
                content = file.read().lower()
                if "loso" in content and "train_test_split" in content:
                    issues.append(f"{f}: Potential LOSO leakage - using random train_test_split instead of Leave-One-Subject-Out.")
                if "dtu" in content and "kul" in content:
                    warnings.append(f"{f}: Both DTU and KUL referenced, ensure no dataset loader confusion.")
                if "sim = einsum" in content and "mean(dim=1)" in content:
                    issues.append(f"{f}: Temporal dimension collapsed before InfoNCE, assuming temporal alignment.")
        except Exception:
            pass
    passed = len(issues) == 0
    memory = search_experiment_memory("EEG Validator " + " ".join(files))
    return json.dumps({"passed": passed, "issues": issues, "warnings": warnings, "memory": memory})

def log_execution(run_dir: Path, message: str):
    log_file = run_dir / "execution.log"
    ts = time.strftime("%H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{ts} {message}\n")

def verify_fix(run_dir: Path, iteration: int, previous_review: dict, current_review: dict, git_diff: str, original_task: str) -> dict:
    prompt = f"""You are an independent verifier. You MUST NOT suggest new code.
Task: {original_task}
Previous Blocking Issues: {previous_review.get('blocking_issues', [])}
Current Diff:
{git_diff}
Current Review Verdict: {current_review.get('verdict')}

Did this diff resolve all previous blocking issues? Were any issues only partially resolved? Were any new issues introduced? Is it safe to accept?
Return ONLY structured JSON:
{{
  "verified": true,
  "confidence": 96,
  "resolved_issues": [],
  "remaining_issues": [],
  "new_issues": [],
  "summary": ""
}}
"""
    raw_response = get_reviewer().review(prompt)
    try:
        json_str = raw_response
        if "```json" in json_str: json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str: json_str = json_str.split("```")[1].split("```")[0].strip()
        verifier_data = json.loads(json_str)
    except Exception as e:
        verifier_data = {"verified": False, "error": str(e), "raw": raw_response}
        
    with open(run_dir / f"verifier_{iteration}.json", "w", encoding="utf-8") as f:
        json.dump(verifier_data, f, indent=2)
        
    return verifier_data

@mcp.tool()
def search_experiment_memory(query: str) -> str:
    """Search previous experiments for lessons and context."""
    repo_root = get_repo_root()
    index_file = repo_root / ".research" / "index.json"
    if not index_file.exists():
        return ""
    
    try:
        with open(index_file, "r", encoding="utf-8") as f:
            index_data = json.load(f)
    except Exception:
        return ""
        
    query_lower = query.lower()
    matches = []
    
    for entry in index_data:
        searchable_text = f"{entry.get('task', '')} {' '.join(entry.get('keywords', []))} {' '.join(entry.get('datasets', []))} {' '.join(entry.get('models', []))} {entry.get('summary', '')}".lower()
        if query_lower in searchable_text or any(word in searchable_text for word in query_lower.split() if len(word) > 3):
            matches.append(entry)
            
    if not matches:
        return ""
        
    out = "PREVIOUS RELATED EXPERIMENTS:\n"
    for m in matches[-3:]:  # Top 3 most recent
        out += f"\nExperiment: {m.get('task')}\n"
        out += f"Status: {m.get('status')}\n"
        out += f"Summary: {m.get('summary')}\n"
        out += "Lessons Learned:\n"
        for l in m.get('lessons', []):
            out += f"- {l}\n"
    return out

def record_research_memory(run_dir: Path, task: str, status: str, final_review: dict):
    repo_root = get_repo_root()
    research_dir = repo_root / ".research"
    experiments_dir = research_dir / "experiments"
    lessons_dir = research_dir / "lessons"
    
    for d in [research_dir, experiments_dir, lessons_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    prompt = f"""Extract research metadata and lessons learned from this autonomous run.
Task: {task}
Status: {status}
Final Review: {json.dumps(final_review)}

Return ONLY strict JSON:
{{
  "summary": "Brief 1-2 sentence summary of what was attempted and the outcome",
  "datasets": ["..."],
  "models": ["..."],
  "keywords": ["..."],
  "lessons": ["Actionable lessons learned, e.g., 'Use subject-level splits for LOSO'"]
}}"""
    
    raw_response = get_reviewer().review(prompt)
    try:
        json_str = raw_response
        if "```json" in json_str: json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str: json_str = json_str.split("```")[1].split("```")[0].strip()
        data = json.loads(json_str)
    except Exception:
        data = {"summary": "Failed to parse metadata", "datasets": [], "models": [], "keywords": [], "lessons": []}
        
    data["experiment_id"] = run_dir.name
    data["task"] = task
    data["status"] = status
    
    with open(experiments_dir / f"{run_dir.name}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        
    index_file = research_dir / "index.json"
    index_data = []
    if index_file.exists():
        try:
            with open(index_file, "r", encoding="utf-8") as f:
                index_data = json.load(f)
        except Exception:
            pass
            
    existing_lessons = set()
    for entry in index_data:
        for l in entry.get("lessons", []):
            existing_lessons.add(l.lower())
            
    unique_lessons = []
    for l in data.get("lessons", []):
        if l.lower() not in existing_lessons:
            unique_lessons.append(l)
            existing_lessons.add(l.lower())
            
    data["lessons"] = unique_lessons
    index_data.append(data)
    
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2)

@mcp.tool()
def autonomous_review(files: list[str], task: str = "", max_iterations: int = 3) -> str:
    """Run an autonomous review loop with Antigravity."""
    repo_root = get_repo_root()
    state_key = str(sorted(files))
    
    if state_key not in _REVIEW_STATES:
        ts_name = time.strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = repo_root / ".runs" / ts_name
        run_dir.mkdir(parents=True, exist_ok=True)
        
        with open(run_dir / "task.md", "w") as f: f.write(task)
        with open(run_dir / "context.md", "w") as f: f.write(str(files))
        
        initial_diff = subprocess.run(["git", "diff"], cwd=repo_root, capture_output=True, text=True).stdout
        if initial_diff.strip():
            with open(run_dir / "initial_diff.patch", "w") as f: f.write(initial_diff)
            
        checkpoint_msg = create_checkpoint(ts_name)
        log_execution(run_dir, "Checkpoint created")
        
        _REVIEW_STATES[state_key] = {
            "iteration": 1, 
            "checkpoint": checkpoint_msg, 
            "prev_blocking": [],
            "prev_review_data": {},
            "run_dir": run_dir,
            "start_time": time.time()
        }
    else:
        _REVIEW_STATES[state_key]["iteration"] += 1
        
    state = _REVIEW_STATES[state_key]
    iteration = state["iteration"]
    checkpoint = state["checkpoint"]
    run_dir = state["run_dir"]
    
    if iteration > 1:
        log_execution(run_dir, "Fix Applied")
    
    static_pass = True
    static_msg = "PASS"
    for f in files:
        target = repo_root / f
        if target.exists() and f.endswith(".py"):
            try:
                subprocess.run(["ruff", "check", "--fix", f], cwd=repo_root, capture_output=True)
            except FileNotFoundError:
                pass
            try:
                comp = subprocess.run(["python", "-m", "py_compile", f], cwd=repo_root, capture_output=True, text=True)
                if comp.returncode != 0:
                    static_pass = False
                    static_msg = f"Compile error in {f}: {comp.stderr}"
                    break
            except FileNotFoundError:
                pass
                
    if not static_pass:
        log_execution(run_dir, "Static Validation FAIL")
        return f"Static validation failed:\n{static_msg}\nFix the syntax/compile errors and call autonomous_review again."
    log_execution(run_dir, "Static Validation PASS")
        
    eeg_res_str = review_experiment(files)
    eeg_data = json.loads(eeg_res_str)
    eeg_pass = eeg_data.get("passed", False)
    with open(run_dir / f"eeg_validation_{iteration}.json", "w") as f:
        json.dump(eeg_data, f, indent=2)
    log_execution(run_dir, f"EEG Validator {'PASS' if eeg_pass else 'FAIL'}")
    
    diff = ""
    if iteration > 1:
        diff_proc = subprocess.run(["git", "diff", "HEAD"], cwd=repo_root, capture_output=True, text=True)
        diff = diff_proc.stdout if diff_proc.stdout else get_git_diff()
        prompt_task = f"{task}\n\nPREVIOUS BLOCKING ISSUES:\n" + "\n".join(state["prev_blocking"])
        prompt_task += f"\n\nCURRENT GIT DIFF:\n{diff}\n\nDid this diff resolve the previous blocking issues? Did it introduce any regressions?"
        raw_review = review_code(files, prompt_task)
    else:
        raw_review = review_code(files, task)
        
    try:
        json_str = raw_review
        if "```json" in json_str: json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str: json_str = json_str.split("```")[1].split("```")[0].strip()
        review_data = json.loads(json_str)
        verdict = review_data.get("verdict", "FAIL")
        conf = int(review_data.get("confidence", 0))
        blocking = review_data.get("blocking_issues", [])
    except Exception as e:
        review_data = {"error": f"Failed to parse JSON: {e}", "raw": raw_review}
        verdict = "FAIL"
        conf = 0
        blocking = ["Failed to parse ChatGPT JSON response."]
        
    with open(run_dir / f"review_{iteration}.json", "w") as f:
        json.dump(review_data, f, indent=2)
        
    chatgpt_pass = (verdict == "PASS" and conf >= 90 and len(blocking) == 0)
    log_execution(run_dir, f"{'Diff Review' if iteration > 1 else 'ChatGPT Review'} {'PASS' if chatgpt_pass else 'FAIL'}")
    
    verifier_pass = True
    if iteration > 1:
        verifier_data = verify_fix(run_dir, iteration, state["prev_review_data"], review_data, diff, task)
        verifier_pass = verifier_data.get("verified", False)
        log_execution(run_dir, f"Independent Verification {'PASS' if verifier_pass else 'FAIL'}")
        
    if static_pass and eeg_pass and chatgpt_pass and verifier_pass:
        final_diff = subprocess.run(["git", "diff", "HEAD"], cwd=repo_root, capture_output=True, text=True).stdout
        with open(run_dir / "final_diff.patch", "w") as f: f.write(final_diff)
        
        runtime = int(time.time() - state["start_time"])
        report = {
            "status": "PASS",
            "iterations": iteration,
            "runtime_seconds": runtime,
            "checkpoint": checkpoint,
            "confidence": conf,
            "verification": "PASS",
            "artifacts": f"./.runs/{run_dir.name}/",
            "final_review": review_data
        }
        with open(run_dir / "final_report.json", "w") as f: json.dump(report, f, indent=2)
        log_execution(run_dir, "Finished")
        record_research_memory(run_dir, task, "PASS", review_data)
        
        del _REVIEW_STATES[state_key]
        return json.dumps(report, indent=2)
        
    if iteration >= max_iterations:
        restore_checkpoint(checkpoint)
        
        runtime = int(time.time() - state["start_time"])
        report = {
            "status": "FAILED_AFTER_MAX_ITERATIONS",
            "iterations": iteration,
            "runtime_seconds": runtime,
            "checkpoint_restored": checkpoint,
            "artifacts": f"./.runs/{run_dir.name}/",
            "last_review": review_data
        }
        with open(run_dir / "final_report.json", "w") as f: json.dump(report, f, indent=2)
        log_execution(run_dir, "Failed after max iterations")
        record_research_memory(run_dir, task, "FAIL", review_data)
        
        del _REVIEW_STATES[state_key]
        return json.dumps(report, indent=2)
        
    state["prev_blocking"] = blocking
    state["prev_review_data"] = review_data
    
    b_str = "\n".join([f"- {i}" for i in blocking])
    w_str = "\n".join([f"- {i}" for i in review_data.get("warnings", [])])
    eeg_b_str = "\n".join([f"- {i}" for i in eeg_data.get("issues", [])])
    
    fix_prompt = f"""Your previous implementation failed review.

Task: {task}

EEG Validator Issues:
{eeg_b_str}

ChatGPT Blocking Issues:
{b_str}

ChatGPT Warnings:
{w_str}

Apply ONLY the required fixes. Do not change unrelated code. Preserve functionality.
After editing, save the files and call autonomous_review again.
"""
    with open(run_dir / f"fix_prompt_{iteration}.md", "w") as f: f.write(fix_prompt)
    
    return fix_prompt

if __name__ == "__main__":
    mcp.run()
