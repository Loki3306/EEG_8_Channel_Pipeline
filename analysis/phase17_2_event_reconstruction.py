import os
import json
import glob
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from decision_engine.context_aware_engine import ContextAwarePolicyEngine

def reconstruct_events():
    csv_files = glob.glob(str(REPO_ROOT / "results" / "phase17_1" / "scenario_streams" / "*.csv"))
    if not csv_files:
        print("No scenario streams found in results/phase17_1/scenario_streams/")
        return
        
    out_dir = REPO_ROOT / "results" / "phase17_2"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "event_log.jsonl"
    
    engine = ContextAwarePolicyEngine(config={
        'confidence_threshold': 0.85,
        'consecutive_required': 10,
        'minimum_switch_gap': 20,
        'cooldown_duration': 40,
        'stabilizing_threshold': 40,
        'minimum_lock_duration': 15,
        'entrenched_confidence': 0.90,
        'entrenched_consecutive': 15
    }, heuristics=['difficulty_adaptation', 'cooldown', 'hysteresis'])
    
    total_events = 0
    with open(out_file, 'w') as f:
        for csv_path in sorted(csv_files):
            df = pd.read_csv(csv_path)
            scenario = Path(csv_path).stem
            engine.reset()
            active_lock = None
            
            for idx, row in df.iterrows():
                res = engine.update(row['prob'], row['margin'])
                
                action = str(res['action'])
                st = str(res['state'])
                
                if action == 'SWITCH_LEFT':
                    active_lock = 1
                elif action == 'SWITCH_RIGHT':
                    active_lock = 0
                elif st in ['UNCERTAIN', 'INITIALIZING', 'WAITING']:
                    active_lock = None
                    
                event = {
                    'scenario': scenario,
                    'timestamp_sec': row['timestamp_sec'],
                    'scene': row['scene_name'],
                    'ground_truth': int(row['ground_truth']),
                    'probability': float(row['prob']),
                    'margin': float(row['margin']),
                    'active_lock': active_lock,
                    'state': st,
                    'action': action,
                    'confidence': float(res.get('confidence', 0.0)),
                    'evidence': float(res.get('evidence', 0.0)),
                    'threshold_used': float(res.get('threshold_used', 0.85)),
                    'consecutive_used': int(res.get('consecutive_used', 10)),
                    'time_in_state': int(engine.time_in_state)
                }
                f.write(json.dumps(event) + '\n')
                total_events += 1
                
    print(f"Reconstructed {total_events} events. Saved to {out_file}")

if __name__ == "__main__":
    print("====================================================")
    print("PHASE 17.2: EVENT RECONSTRUCTION")
    print("====================================================")
    reconstruct_events()
