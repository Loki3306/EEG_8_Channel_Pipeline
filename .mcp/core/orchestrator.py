import sys
import json
import subprocess
import time
from pathlib import Path
from .plugins import detect_project_plugin
from .memory import search_memory
from .context import gather_context
from .browser import requires_external_research, perform_external_research
from .planner import generate_plan

_ACTIVE_TASK = {}

def start_task_impl(task: str, repo_root: Path) -> dict:
    global _ACTIVE_TASK
    
    # 1. Determine task type
    task_type = "implementation"
    t_lower = task.lower()
    is_bug_fix = any(x in t_lower for x in ["debug", "fix", "error", "bug"])
    is_research = any(x in t_lower for x in ["research", "paper", "concept"])
    is_analysis = any(x in t_lower for x in ["analyze", "metric", "result"])
    is_refactor = any(x in t_lower for x in ["refactor", "cleanup", "rewrite"])
    
    if is_bug_fix:
        task_type = "debugging"
    elif is_research:
        task_type = "research"
    elif is_analysis:
        task_type = "analysis"
    elif is_refactor:
        task_type = "refactoring"
        
    # 2. Project & Plugin Detection
    plugin = detect_project_plugin(repo_root)
    
    # 3. Load Memory & Previous Experiments
    mem = search_memory(repo_root, task)
    
    # 4. Gather Repository Context
    ctx = gather_context(repo_root, task)
    
    # 5. External Research Decision (Part 7: Context Selection)
    ext_research = ""
    if is_research or requires_external_research(task):
        ext_research = perform_external_research(task)
        
    # 6. ChatGPT Decision (Part 7: Context Selection)
    # Only use Browser ChatGPT when deep reasoning or architectural planning is required
    requires_chatgpt = is_research or is_refactor or "architect" in t_lower or "matchnet" in t_lower
    
    if requires_chatgpt:
        plan = generate_plan(task, ctx, mem)
    else:
        # Avoid unnecessary tool invocation
        plan = f"Fast Plan: Proceed with local repository edits. Affected files: {ctx.get('related_files', [])}"
    
    # Setup Planning State
    _ACTIVE_TASK = {
        "task": task,
        "task_type": task_type,
        "project": repo_root.name,
        "plugin": plugin.name(),
        "context": ctx,
        "memory": mem,
        "plan": plan,
        "phase": "PLANNING",
        "start_time": time.time()
    }
    
    plan_details = {
        "status": "READY",
        "phase": "PLANNING",
        "requires_user_approval": True,
        "project": repo_root.name,
        "task_type": task_type,
        "summary": f"Orchestrating task: {task}",
        "repository_context": ctx.get("repository_summary")[:500] + "..." if ctx.get("repository_summary") else "",
        "previous_experiments": mem.get("experiments"),
        "related_files": ctx.get("related_files"),
        "implementation_plan": plan,
        "risks": ["Subject leakage in LOSO splits" if plugin.name() == "EEG" else "General deployment risks"],
        "estimated_files_to_modify": ["models/matchnet.py", "scratch/flawed_eeg.py"] if plugin.name() == "EEG" else []
    }
    
    return {
        "status": "WAITING_FOR_APPROVAL",
        "plan": plan_details
    }

def continue_task_impl(repo_root: Path) -> str:
    global _ACTIVE_TASK
    import server
    
    if not _ACTIVE_TASK or _ACTIVE_TASK.get("phase") != "PLANNING":
        return json.dumps({"error": "No task in PLANNING phase. Run start_task first."})
        
    _ACTIVE_TASK["phase"] = "EXECUTION"
    
    ts_name = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = repo_root / ".runs" / ts_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    with open(run_dir / "task.md", "w") as f: f.write(_ACTIVE_TASK["task"])
    with open(run_dir / "context.md", "w") as f: f.write(str(_ACTIVE_TASK["context"].get("related_files")))
    
    checkpoint_msg = server.create_checkpoint(ts_name)
    server.log_execution(run_dir, "Checkpoint created")
    
    _ACTIVE_TASK["run_dir"] = run_dir
    _ACTIVE_TASK["checkpoint"] = checkpoint_msg
    
    state_key = str(sorted(_ACTIVE_TASK["context"].get("related_files", [])))
    server._REVIEW_STATES[state_key] = {
        "iteration": 1,
        "checkpoint": checkpoint_msg,
        "prev_blocking": [],
        "prev_review_data": {},
        "run_dir": run_dir,
        "start_time": _ACTIVE_TASK["start_time"]
    }
    
    return json.dumps({
        "status": "EXECUTION_STARTED",
        "phase": "EXECUTION",
        "checkpoint": checkpoint_msg,
        "run_directory": str(run_dir.relative_to(repo_root))
    }, indent=2)

def finish_task_impl(repo_root: Path) -> str:
    global _ACTIVE_TASK
    import server
    
    if not _ACTIVE_TASK or _ACTIVE_TASK.get("phase") != "EXECUTION":
        return json.dumps({"error": "No task in EXECUTION phase. Run continue_task first."})
        
    try:
        res = subprocess.run(["git", "diff", "--name-only"], cwd=repo_root, capture_output=True, text=True)
        files = [f.strip() for f in res.stdout.split('\n') if f.strip()]
    except Exception:
        files = []
        
    if not files:
        files = _ACTIVE_TASK.get("context", {}).get("related_files", [])
        
    task_desc = _ACTIVE_TASK.get("task", "Verify changes")
    review_output = server.autonomous_review(files, task_desc)
    
    if "PASS" in review_output:
        _ACTIVE_TASK = {}
        
    return review_output

def agent_impl(task: str, repo_root: Path) -> str:
    global _ACTIVE_TASK
    
    # Check if the user is providing approval to proceed
    is_approval = task.strip().lower() in ["yes", "go", "proceed", "implement", "approved"]
    
    if is_approval and _ACTIVE_TASK and _ACTIVE_TASK.get("phase") == "PLANNING":
        return continue_task_impl(repo_root)
        
    # Otherwise, it is a new task initiation
    res = start_task_impl(task, repo_root)
    return json.dumps(res, indent=2)
