"""
Verify Raw Trial Extraction
Loads S1.mat (raw) and S1_data_preproc.mat (preproc) to map 
raw triggers to the 60 preprocessed trials and verify reconstruction.
"""
import scipy.io as sio
import numpy as np
from pathlib import Path
import sys

def print_and_write(f_out, text):
    print(text)
    f_out.write(text + "\n")

raw_path = Path("/kaggle/input/datasets/lowk1ee/raw-eeh/S1.mat")
preproc_path = Path("/kaggle/input/dtu-eeg-data/S1_data_preproc.mat")

out_dir = Path("/kaggle/working/reports")
out_dir.mkdir(parents=True, exist_ok=True)
report_path = out_dir / "trigger_alignment_report.md"

with open(report_path, "w") as f_out:
    print_and_write(f_out, "# Raw Trigger & Trial Alignment Audit\n")
    
    # 1. Inspect Preprocessed Data
    print_and_write(f_out, "## 1. Preprocessed Data Exploration")
    if not preproc_path.exists():
        print_and_write(f_out, f"ERROR: Preprocessed path {preproc_path} not found.")
    else:
        try:
            mat_pre = sio.loadmat(preproc_path, squeeze_me=True, struct_as_record=False)
            pre_keys = [k for k in mat_pre.keys() if not k.startswith("__")]
            print_and_write(f_out, f"Keys in Preprocessed MAT: {pre_keys}")
            
            data_pre = mat_pre.get('data')
            if data_pre:
                if hasattr(data_pre, 'eeg'):
                    print_and_write(f_out, f"Preprocessed eeg shape: {data_pre.eeg.shape}")
                    num_trials = data_pre.eeg.shape[1]
                    print_and_write(f_out, f"Number of Preprocessed Trials: {num_trials}")
                else:
                    print_and_write(f_out, "No 'eeg' attribute in preproc data struct.")
            
            # Check for expinfo
            if 'expinfo' in mat_pre:
                expinfo = mat_pre['expinfo']
                print_and_write(f_out, "Found 'expinfo' in Preprocessed MAT!")
                print_and_write(f_out, f"expinfo type: {type(expinfo)}")
                if hasattr(expinfo, 'dtype'):
                    print_and_write(f_out, f"expinfo dtype fields: {expinfo.dtype.names}")
                elif isinstance(expinfo, np.ndarray) and len(expinfo) > 0 and hasattr(expinfo[0], 'trigger'):
                    print_and_write(f_out, "expinfo is array of structs.")
            else:
                print_and_write(f_out, "No 'expinfo' at root level of Preprocessed MAT.")
                if data_pre and hasattr(data_pre, 'expinfo'):
                    print_and_write(f_out, "Found 'expinfo' inside data struct!")
                    
        except Exception as e:
            print_and_write(f_out, f"Failed to load Preprocessed MAT: {e}")
            
    print_and_write(f_out, "\n## 2. Raw Data Exploration")
    if not raw_path.exists():
        print_and_write(f_out, f"ERROR: Raw path {raw_path} not found.")
    else:
        try:
            mat_raw = sio.loadmat(raw_path, squeeze_me=True, struct_as_record=False)
            raw_keys = [k for k in mat_raw.keys() if not k.startswith("__")]
            print_and_write(f_out, f"Keys in Raw MAT: {raw_keys}")
            
            if 'expinfo' in mat_raw:
                print_and_write(f_out, "Found 'expinfo' in Raw MAT!")
                
            data_raw = mat_raw.get('data')
            if data_raw:
                if hasattr(data_raw, 'event') and hasattr(data_raw.event, 'eeg'):
                    events_val = data_raw.event.eeg.value
                    events_sample = data_raw.event.eeg.sample
                    
                    print_and_write(f_out, f"\nTotal Triggers: {len(events_val)}")
                    
                    # Convert to list for easier processing
                    vals = events_val.tolist() if hasattr(events_val, 'tolist') else list(events_val)
                    samples = events_sample.tolist() if hasattr(events_sample, 'tolist') else list(events_sample)
                    
                    # Look for sequence of trial triggers vs 191
                    print_and_write(f_out, "\nFull sequence of triggers (Index: Value @ Sample):")
                    for i in range(len(vals)):
                        print_and_write(f_out, f"[{i:3d}] Trigger {vals[i]:3d} @ {samples[i]:8d}")
                        
        except Exception as e:
            print_and_write(f_out, f"Failed to load Raw MAT: {e}")
            
print(f"\nAudit complete. Report saved to {report_path}")
