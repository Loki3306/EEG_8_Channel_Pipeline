import sys
import json
import pickle
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Support both local paths and Kaggle paths
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Fetch data directory logic from ridge_aad
try:
    from baselines.ridge_aad import load_subject_examples, DATA_DIR
except ImportError:
    # Fallback if run directly and paths are messy
    import os
    if "EEG_DATA_DIR" in os.environ:
        DATA_DIR = Path(os.environ["EEG_DATA_DIR"])
    elif Path("/kaggle/input").exists():
        mat_files = list(Path("/kaggle/input").rglob("S*_data_preproc.mat"))
        if mat_files:
            DATA_DIR = mat_files[0].parent
        else:
            DATA_DIR = Path("/kaggle/input")
    else:
        DATA_DIR = Path("C:/Users/lokes/Downloads/archive (2)/DATA_preproc")
        
    from baselines.ridge_aad import load_subject_examples

FS = 64
WINDOW_SEC = 10

def get_mapping_data():
    # If on Kaggle, the pkl might be in the input dataset instead of REPO_ROOT/data
    if Path("/kaggle/input").exists():
        env_files = list(Path("/kaggle/input").rglob("gammatone_envelopes.pkl"))
        if env_files:
            env_file = env_files[0]
        else:
            env_file = REPO_ROOT / "data" / "gammatone_envelopes.pkl"
            
        map_files = list(Path("/kaggle/input").rglob("audio_mapping.json"))
        if map_files:
            map_file = map_files[0]
        else:
            map_file = REPO_ROOT / "data" / "audio_mapping.json"
    else:
        map_file = REPO_ROOT / "data" / "audio_mapping.json"
        env_file = REPO_ROOT / "data" / "gammatone_envelopes.pkl"
        
    with open(map_file, 'r') as f:
        mapping = json.load(f)
    with open(env_file, 'rb') as f:
        envelopes = pickle.load(f)
    return mapping, envelopes

def extract_features(wav, env_gammatone):
    rms = np.sqrt(np.mean(wav**2))
    var = np.var(wav)
    peak = np.max(np.abs(wav))
    
    gt_mean = np.mean(env_gammatone, axis=1) # 28 features
    gt_var = np.var(env_gammatone, axis=1)   # 28 features
    
    return np.concatenate([[rms, var, peak], gt_mean, gt_var])

def run_audio_audit():
    print("Loading audio mapping and envelopes...")
    mapping, envelopes = get_mapping_data()
    print(f"Data directory identified as: {DATA_DIR}")
    
    subjects_to_test = ["S8", "S10", "S11", "S12", "S16"]
    
    print("\n--- Audio-Only Separability Audit (10s window) ---")
    
    for sub in subjects_to_test:
        mat_path = DATA_DIR / f"{sub}_data_preproc.mat"
        if not mat_path.exists():
            print(f"Skipping {sub}: file not found at {mat_path}")
            continue
            
        examples = load_subject_examples(mat_path)
        
        X = []
        y = []
        
        for i, ex in enumerate(examples):
            trial_key = f"trial_{i}"
            
            # Need to map trial_key correctly because subject ids usually don't include _data_preproc in mapping.json
            sub_key = sub
            if sub_key in mapping and trial_key in mapping[sub_key]:
                fname_a = mapping[sub_key][trial_key]["wavA"]["filename"]
                fname_b = mapping[sub_key][trial_key]["wavB"]["filename"]
                env_a_full = envelopes[fname_a] 
                env_b_full = envelopes[fname_b] 
            else:
                continue
                
            wav_a_full = ex.wav_a
            wav_b_full = ex.wav_b
            
            min_len = min(len(wav_a_full), env_a_full.shape[1])
            
            win_samples = WINDOW_SEC * FS
            start = 0
            
            while start + win_samples <= min_len:
                end = start + win_samples
                
                w_a = wav_a_full[start:end]
                w_b = wav_b_full[start:end]
                
                e_a = env_a_full[:, start:end]
                e_b = env_b_full[:, start:end]
                
                feat_a = extract_features(w_a, e_a)
                feat_b = extract_features(w_b, e_b)
                
                # Model learns mapping from Feature Difference to Label
                diff_features = feat_a - feat_b
                X.append(diff_features)
                
                # Label 1 if A attended, 0 if B attended
                if ex.label == 1:
                    y.append(1)
                else:
                    y.append(0)
                    
                start += win_samples
                
        if len(X) == 0:
            print(f"{sub}: No valid trials extracted.")
            continue
            
        X = np.array(X)
        y = np.array(y)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        clf = LogisticRegression(max_iter=1000)
        scores = cross_val_score(clf, X_scaled, y, cv=5)
        
        print(f"{sub} Audio-Only Accuracy: {scores.mean()*100:.2f}% (Std: {scores.std()*100:.2f}%)")

if __name__ == "__main__":
    run_audio_audit()
