import ast
import astor

file_path = ".mcp/server.py"

with open(file_path, "r", encoding="utf-8") as f:
    source = f.read()

tree = ast.parse(source)

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

low_level_msg = "Low-level repository helper. Prefer the high-level agent() tool for almost all engineering tasks.\n\n"
agent_msg = "Primary entry point for engineering. Handles repository understanding, project memory, experiment memory, implementation planning, browser research, Browser ChatGPT reasoning, Repository CI, autonomous review, independent verification, and experiment tracking. Use this tool for all non-trivial engineering tasks.\n\n"

for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef):
        if node.name in low_level_funcs:
            doc = ast.get_docstring(node)
            if doc and "Low-level repository helper" not in doc:
                new_doc = low_level_msg + doc.replace('Low-level repository helper. Prefer the high-level agent() tool for almost all engineering tasks."""', '').strip()
                node.body[0].value.value = new_doc
            elif not doc:
                node.body.insert(0, ast.Expr(value=ast.Constant(value=low_level_msg.strip())))
        elif node.name == "agent":
            doc = ast.get_docstring(node)
            if doc and "Primary entry point" not in doc:
                new_doc = agent_msg + doc.replace('Primary entry point for engineering.', '').strip()
                node.body[0].value.value = new_doc

with open(file_path, "w", encoding="utf-8") as f:
    f.write(astor.to_source(tree))

print("Fixed docstrings using astor.")
