import json
from pathlib import Path
from baselines.ridge_aad import load_subject_examples, subject_files

def main():
    paths = subject_files()
    if not paths:
        print("No subjects found")
        return
        
    trial_labels = {}
    
    for p in paths:
        exs = load_subject_examples(p)
        for i, ex in enumerate(exs):
            if i not in trial_labels:
                trial_labels[i] = []
            trial_labels[i].append(ex.label)
            
    print("Trial | Labels across subjects")
    print("-" * 30)
    for i in range(len(trial_labels)):
        unique_labels = set(trial_labels[i])
        print(f"Trial {i:2d} | Unique Labels: {unique_labels} | All: {trial_labels[i][:5]}...")
        if len(unique_labels) == 1:
            print(f"  [!] LEAKAGE! Trial {i} always has label {list(unique_labels)[0]}")
            
if __name__ == "__main__":
    main()
