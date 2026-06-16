import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_WORDS = [
    r'\brandom\b',
    r'\bmock\b',
    r'\bdummy\b',
    r'\bplaceholder\b',
    r'\bTODO\b',
    r'\bTBD\b'
]

def audit_repository():
    compiled_regexes = [(word, re.compile(word, re.IGNORECASE)) for word in FORBIDDEN_WORDS]
    
    report_path = REPO_ROOT / "results" / "reports" / "research_integrity_audit.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    violations = []
    
    # Walk through the repository
    for root, dirs, files in os.walk(REPO_ROOT):
        # Exclude results and .git
        if '.git' in root or 'results' in root:
            continue
            
        for file in files:
            if not (file.endswith('.py') or file.endswith('.md')):
                continue
                
            filepath = Path(root) / file
            
            # Skip the audit script itself and rules file
            if file == "audit_integrity.py" or file == "RESEARCH_RULES.md":
                continue
                
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        for word, regex in compiled_regexes:
                            if regex.search(line):
                                rel_path = filepath.relative_to(REPO_ROOT)
                                violations.append((str(rel_path), line_num, word, line.strip()))
            except Exception as e:
                print(f"Failed to read {filepath}: {e}")
                
    # Write report
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Research Integrity Audit Report\n\n")
        
        if not violations:
            f.write("**Status:** Clean. No forbidden terminology or placeholders found.\n")
        else:
            f.write(f"**Status:** {len(violations)} violations found.\n\n")
            f.write("The following files contain forbidden words that violate `RESEARCH_RULES.md`:\n\n")
            f.write("| File | Line | Forbidden Keyword | Context |\n")
            f.write("|---|---|---|---|\n")
            for rel_path, line_num, word, ctx in violations:
                # Truncate context if too long
                if len(ctx) > 100:
                    ctx = ctx[:97] + "..."
                # Escape pipes for markdown table
                ctx = ctx.replace('|', '\\|')
                f.write(f"| `{rel_path}` | {line_num} | `{word.replace(r'\\b', '')}` | `{ctx}` |\n")
                
    print(f"Audit complete. Report generated at {report_path.relative_to(REPO_ROOT)}")

if __name__ == "__main__":
    audit_repository()
