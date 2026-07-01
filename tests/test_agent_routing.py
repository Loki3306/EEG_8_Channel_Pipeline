import sys
from pathlib import Path
import json

# Add project root to sys.path to resolve internal MCP imports
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / ".mcp"))

from core.orchestrator import agent_impl

def test_ml_engineering_routing():
    """
    Simulates a typical ML implementation request and verifies that the agent()
    orchestrator correctly classifies it as a Machine Learning Engineering task
    and initiates the full pipeline.
    """
    prompt = "Implement confidence-aware MatchNet for KUL."
    print(f"Testing Prompt: '{prompt}'")
    
    # Execute the agent orchestrator logic directly
    result_json = agent_impl(prompt, repo_root)
    result = json.loads(result_json)
    
    # 1. Assert it entered the orchestrator waiting for approval
    assert result.get("status") == "WAITING_FOR_APPROVAL", "Did not enter Planning Phase."
    
    plan = result.get("plan", {})
    task_type = plan.get("task_type")
    
    # 2. Assert it classified as a Machine Learning / Research task
    assert task_type == "research", f"Expected 'research' task_type for ML task, got '{task_type}'"
    
    # 3. Assert the Planning Context contains the execution-first structure
    planning_context = plan.get("planning_context", "")
    assert "Planning Context" in planning_context, "Missing 'Planning Context' header"
    assert "Implementation Constraints:" in planning_context, "Missing Implementation Constraints in planning context"
    assert "Validation Plan:" in planning_context, "Missing Validation Plan in planning context"
    assert "Do not invent new architectures" in planning_context, "Missing explicit redesign constraint"
    
    print("Integration Test Passed: Prompt correctly routed through agent() and produced the authoritative Planning Context.")

if __name__ == "__main__":
    test_ml_engineering_routing()
