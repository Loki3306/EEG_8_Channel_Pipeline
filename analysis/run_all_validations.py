import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SCRIPTS = [
    "analysis/run1_reproducibility.py",
    "analysis/run1_window_scaling.py",
    "analysis/run1_calibration.py",
    "analysis/run1_statistics.py",
    "analysis/run1_subject_stability.py",
    "analysis/run1_training_dynamics.py"
]

def main():
    print("="*60)
    print("Starting Comprehensive Scientific Validation Suite (Run 1)")
    print("="*60)
    
    for script in SCRIPTS:
        script_path = REPO_ROOT / script
        if not script_path.exists():
            print(f"ERROR: Script not found: {script_path}")
            continue
            
        print(f"\n>>> Running {script} ...")
        result = subprocess.run(["python", str(script_path)], cwd=str(REPO_ROOT))
        
        if result.returncode != 0:
            print(f"!!! Error running {script} (Code: {result.returncode})")
        else:
            print(f">>> Finished {script}")
            
    print("\n" + "="*60)
    print("Validation Suite Completed.")
    print("Please download the 'results/conformer_loso' directory from Kaggle.")
    print("="*60)

if __name__ == "__main__":
    main()
