import re

file_path = ".mcp/server.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Define the low-level functions
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

low_level_msg = "Low-level repository helper. Prefer the high-level agent() tool for almost all engineering tasks."
agent_msg = "Primary entry point for engineering. Handles repository understanding, project memory, experiment memory, implementation planning, browser research, Browser ChatGPT reasoning, Repository CI, autonomous review, independent verification, and experiment tracking. Use this tool for all non-trivial engineering tasks."

for func in low_level_funcs:
    # Find def func():\n    """Docstring"""
    pattern = rf'(def {func}\([^)]*\)(?:\s*->\s*[^:]+)?:.*?""")'
    def repl_low_level(m):
        doc = m.group(1)
        if "Low-level repository helper" not in doc:
            return doc[:-3] + f" {low_level_msg}\"\"\""
        return doc
        
    content = re.sub(pattern, repl_low_level, content, flags=re.DOTALL)

# Update agent()
pattern_agent = r'(def agent\([^)]*\)(?:\s*->\s*[^:]+)?:.*?""")'
def repl_agent(m):
    doc = m.group(1)
    if "Primary entry point" not in doc:
        return doc[:-3] + f" {agent_msg}\"\"\""
    return doc

content = re.sub(pattern_agent, repl_agent, content, flags=re.DOTALL)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Updated server.py successfully.")
