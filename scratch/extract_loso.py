import json
import re

log_path = r'C:\Users\lokes\.gemini\antigravity-ide\brain\75bbcbaa-6f1a-4646-8451-13a8c62e2b4c\.system_generated\logs\transcript.jsonl'
output = ""

with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        if 'Accuracy Normal' in line and 'Evaluating fold' in line:
            data = json.loads(line)
            content = data.get('content', '')
            # Extract subjects and their accuracies
            matches = re.findall(r'Evaluating fold with held-out subject: (S\d+_data_preproc).*?Accuracy Normal\s*:\s*([\d\.]+)%', content, re.DOTALL)
            for m in matches:
                output += f"{m[0]}: {m[1]}%\n"

# Deduplicate
lines = list(set(output.strip().split('\n')))
for l in sorted(lines):
    print(l)
