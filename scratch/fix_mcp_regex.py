import sys
import re

file_path = ".mcp/server.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# For low level funcs, we replace the first """ (or ''') inside the function definition.
low_level_funcs = [
    "search_project",
    "search_experiment_memory",
    "project_summary",
    "repository_statistics",
    "find_dataset",
    "find_model_definition",
    "review_code",
    "autonomous_review",
    "start_task",
    "continue_task",
    "finish_task"
]

msg = "Low-level repository helper. Prefer the high-level agent() tool for almost all engineering tasks. "

for func in low_level_funcs:
    # Match: def func(...):[whitespace]"""[docstring]"""
    # We want to insert msg right after the first """
    pattern = rf'(def {func}\([^)]*\)(?:\s*->\s*[^:]+)?:[\s\n]*\"\"\")'
    content = re.sub(pattern, rf'\1{msg}', content)

agent_msg = "Primary entry point for engineering. Handles repository understanding, project memory, experiment memory, implementation planning, browser research, Browser ChatGPT reasoning, Repository CI, autonomous review, independent verification, and experiment tracking. Use this tool for all non-trivial engineering tasks. "

pattern_agent = r'(def agent\([^)]*\)(?:\s*->\s*[^:]+)?:[\s\n]*\"\"\")'
content = re.sub(pattern_agent, rf'\1{agent_msg}', content)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Success")
