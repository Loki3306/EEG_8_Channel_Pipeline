import pandas as pd
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

def generate_case_studies():
    out_dir = REPO_ROOT / "results" / "phase17_2"
    
    try:
        switches_df = pd.read_csv(out_dir / "switch_events.csv")
    except FileNotFoundError:
        print("switch_events.csv not found")
        return
        
    print("\n====================================================")
    print("MANUAL CASE STUDIES (RAW DATA)")
    print("====================================================\n")
    
    true_switches = switches_df[switches_df['is_correct'] == True]
    false_switches = switches_df[switches_df['is_correct'] == False]
    
    print("### CASE STUDY 1: A Successful True Switch\n")
    if not true_switches.empty:
        # Pick one true switch (e.g., from scenario 5)
        ts = true_switches.iloc[0]
        print(f"**Scenario**: {ts['scenario']}")
        print(f"**Timestamp**: {ts['timestamp_sec']}s")
        print(f"**Action**: Switched from {ts['old_lock']} to {ts['new_lock']}")
        print(f"**Ground Truth**: {ts['ground_truth']}")
        print(f"**Margin at Switch**: {ts['margin']:.3f}\n")
    else:
        print("No True Switches found.\n")
        
    print("### CASE STUDY 2: A False Switch (Low Margin)\n")
    low_margin_fs = false_switches[false_switches['margin'] < 0.15]
    if not low_margin_fs.empty:
        fs = low_margin_fs.iloc[0]
        print(f"**Scenario**: {fs['scenario']}")
        print(f"**Timestamp**: {fs['timestamp_sec']}s")
        print(f"**Action**: False Switch from {fs['old_lock']} to {fs['new_lock']}")
        print(f"**Ground Truth**: {fs['ground_truth']}")
        print(f"**Margin at Switch**: {fs['margin']:.3f}\n")
    else:
        print("No Low-Margin False Switches found.\n")
        
    print("### CASE STUDY 3: A False Switch (Thrashing/Oscillation)\n")
    if len(false_switches) > 1:
        fs2 = false_switches.iloc[-1]
        print(f"**Scenario**: {fs2['scenario']}")
        print(f"**Timestamp**: {fs2['timestamp_sec']}s")
        print(f"**Action**: False Switch from {fs2['old_lock']} to {fs2['new_lock']}")
        print(f"**Ground Truth**: {fs2['ground_truth']}")
        print(f"**Margin at Switch**: {fs2['margin']:.3f}\n")
    
if __name__ == "__main__":
    generate_case_studies()
