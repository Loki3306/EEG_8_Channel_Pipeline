import sys
from pathlib import Path

core_dir = Path(__file__).resolve().parent
parent_dir = core_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

def generate_plan(task: str, context: dict, memory: dict) -> str:
    import server
    
    prompt = f"""You are a senior software architect. Generate a concise implementation plan for the following task.
Do NOT write code. Just generate the plan.

Task: {task}

Repository Context:
- Related Files: {context.get('related_files', [])}
- Datasets: {context.get('datasets', [])}

Research Memory Lessons:
{chr(10).join('- ' + l for l in memory.get('lessons', []))}

Format the plan with the following sections:
## Implementation Plan
1. Affected Files
2. Reusable Components
3. Implementation Strategy
4. Risks
"""
    try:
        reviewer = server.get_reviewer()
        plan = reviewer.review(prompt)
        return plan
    except Exception as e:
        return f"Failed to generate plan via ChatGPT: {e}\nFallback: Proceed with incremental changes focusing on LOSO splits and symmetric InfoNCE objective."
