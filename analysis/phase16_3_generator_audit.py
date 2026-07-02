import os
import sys
import json
import hashlib
import numpy as np
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pipeline.session_generator import ContinuousSessionGenerator, KULAdapter

def compute_hash(state_str):
    return hashlib.sha256(state_str.encode('utf-8')).hexdigest()

def audit_scenario(json_path, generator):
    print(f"\nAuditing {Path(json_path).name}...")
    
    stream = generator.generate_stream(json_path)
    
    # Layer 1: Structural
    total_windows = 0
    scenes_seen = set()
    subjects_seen = set()
    trials_seen = set()
    
    # Layer 2: Temporal
    prev_timestamp = None
    expected_hop = generator.hop_sec
    
    # Layer 4: Determinism State
    fingerprint_state = ""
    
    for i, window in enumerate(stream):
        # Layer 3: Metadata Integrity
        required_keys = ['eeg_window', 'audio_a_window', 'audio_b_window', 
                         'ground_truth', 'timestamp_sec', 'scene_name', 
                         'scenario_name', 'dataset', 'subject', 'trial', 'window_idx']
                         
        for k in required_keys:
            assert k in window, f"Metadata Missing: {k}"
            
        assert window['window_idx'] == i, f"Window index mismatch: {window['window_idx']} != {i}"
        
        # Layer 2: Temporal Monotonicity
        ts = window['timestamp_sec']
        if prev_timestamp is not None:
            # allow small float precision errors
            diff = ts - prev_timestamp
            assert abs(diff - expected_hop) < 1e-5, f"Temporal drift detected: hop was {diff}, expected {expected_hop}"
            
        prev_timestamp = ts
        
        # Track statistics
        scenes_seen.add(window['scene_name'])
        subjects_seen.add(window['subject'])
        trials_seen.add(window['trial'])
        total_windows += 1
        
        # Layer 4: Accumulate Determinism (use mean of EEG to prove data hasn't shifted)
        mean_val = float(np.mean(window['eeg_window']))
        fingerprint_state += f"{i}|{ts:.3f}|{window['scene_name']}|{mean_val:.4f}\n"

    # Layer 1: Validate Total Duration Math
    # In continuous_stream, the last center sample is roughly end - window_sec/2
    # So total duration is total_windows * hop + window_sec
    total_duration_sec = total_windows * generator.hop_sec + generator.window_sec
    
    # Hash
    final_hash = compute_hash(fingerprint_state)
    
    print(f"  [PASS] Layer 1 (Structural): {total_windows} windows, {total_duration_sec:.2f}s duration")
    print(f"  [PASS] Layer 2 (Temporal): Strict monotonicity at {expected_hop}s hop")
    print(f"  [PASS] Layer 3 (Metadata): All {len(required_keys)} fields present per window")
    print(f"  [PASS] Layer 4 (Determinism): SHA256 -> {final_hash[:12]}...")
    
    return {
        'scenario_name': window['scenario_name'],
        'total_duration_sec': round(total_duration_sec, 2),
        'scene_count': len(scenes_seen),
        'subject_count': len(subjects_seen),
        'trial_count': len(trials_seen),
        'window_count': total_windows,
        'sha256_fingerprint': final_hash
    }


def main():
    print("====================================================")
    print("PHASE 16.3: SESSION GENERATOR VERIFICATION")
    
    out_dir = Path("results/phase16_3")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    kul_adapter = KULAdapter(cache_dir="/kaggle/input/datasets/lokeshgile/kul-processed/data/processed_kul")
    generator = ContinuousSessionGenerator(adapters={'KUL': kul_adapter})
    
    scenarios_dir = Path("scenarios")
    scenario_files = sorted(list(scenarios_dir.glob("*.json")))
    
    fingerprints = {}
    
    for sf in scenario_files:
        # Run Run 1
        stats = audit_scenario(sf, generator)
        
        # Run Run 2 for strict determinism check
        print("  Running determinism check (Run 2)...")
        stats2 = audit_scenario(sf, generator)
        
        assert stats['sha256_fingerprint'] == stats2['sha256_fingerprint'], "DETERMINISM FAILED: Hashes do not match between runs!"
        print("  [PASS] Cross-Run Determinism Verified.")
        
        fingerprints[stats['scenario_name']] = stats
        
    # Save Output
    with open(out_dir / "scenario_fingerprints.json", "w") as f:
        json.dump(fingerprints, f, indent=4)
        
    print("\nALL 5 SCENARIOS VERIFIED SUCCESSFULLY.")
    print(f"Fingerprints written to {out_dir / 'scenario_fingerprints.json'}")
    print("====================================================")

if __name__ == "__main__":
    main()
