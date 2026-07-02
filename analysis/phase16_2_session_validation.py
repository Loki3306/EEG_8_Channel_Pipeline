import os
import sys
import json
import pandas as pd
from pathlib import Path
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pipeline.session_generator import ContinuousSessionGenerator, KULAdapter

def validate_scenario(json_path, generator):
    print(f"Validating {Path(json_path).name}...")
    
    stream = generator.generate_stream(json_path)
    
    total_windows = 0
    scenes_seen = []
    current_scene = None
    
    start_time = time.time()
    
    for window in stream:
        total_windows += 1
        scene_name = window['scene_name']
        
        if scene_name != current_scene:
            scenes_seen.append({
                'scene_name': scene_name,
                'start_window_idx': window['window_idx'],
                'start_timestamp_sec': window['timestamp_sec'],
                'ground_truth': window['ground_truth']
            })
            current_scene = scene_name
            
    elapsed = time.time() - start_time
    
    # Calculate duration (assuming 50ms hop)
    # duration_sec = total_windows * 0.05 + 2.0 (approx)
    total_duration_sec = total_windows * generator.hop_sec + generator.window_sec
    
    return {
        'scenario_name': scenes_seen[0]['scene_name'] if scenes_seen else "Unknown", 
        'total_windows': total_windows,
        'total_duration_sec': round(total_duration_sec, 2),
        'scenes': scenes_seen,
        'generation_time_sec': round(elapsed, 2)
    }

def main():
    print("====================================================")
    print("PHASE 16.2")
    
    out_dir = Path("results/phase16_2")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Initializing Generator and loading KULAdapter...")
    kul_adapter = KULAdapter(cache_dir="/kaggle/input/datasets/lokeshgile/kul-processed/data/processed_kul")
    generator = ContinuousSessionGenerator(adapters={'KUL': kul_adapter})
    
    scenarios_dir = Path("scenarios")
    scenario_files = sorted(list(scenarios_dir.glob("*.json")))
    
    if not scenario_files:
        print("No scenarios found!")
        return
        
    print("Scenario Parsing ........ DONE")
    
    all_summaries = {}
    all_scenes = []
    
    total_windows_all = 0
    total_duration_all = 0
    
    for sf in scenario_files:
        # Load scenario json just for the scenario name
        with open(sf, 'r') as f:
            scen_json = json.load(f)
        scen_name = scen_json['scenario_name']
        
        try:
            stats = validate_scenario(sf, generator)
            stats['scenario_name'] = scen_name
            
            all_summaries[scen_name] = {
                'total_windows': stats['total_windows'],
                'total_duration_sec': stats['total_duration_sec'],
                'scene_count': len(stats['scenes']),
                'generation_time_sec': stats['generation_time_sec']
            }
            
            for s in stats['scenes']:
                s['scenario_name'] = scen_name
                all_scenes.append(s)
                
            total_windows_all += stats['total_windows']
            total_duration_all += stats['total_duration_sec']
            
        except Exception as e:
            print(f"Error processing {sf.name}: {e}")
            
    print("Continuous Stream ....... DONE")
    
    # Save outputs
    with open(out_dir / "continuous_stream_statistics.json", "w") as f:
        json.dump(all_summaries, f, indent=4)
        
    df_scenes = pd.DataFrame(all_scenes)
    df_scenes.to_csv(out_dir / "scene_timeline.csv", index=False)
    
    print("Validation .............. DONE")
    print("Files Written")
    
    print(f"Scenario Count : {len(scenario_files)}")
    print(f"Scene Count    : {len(all_scenes)}")
    print(f"Total Windows  : {total_windows_all}")
    print(f"Total Duration : {round(total_duration_all / 60, 2)} minutes")
    print("Done")
    print("====================================================")

if __name__ == "__main__":
    main()
