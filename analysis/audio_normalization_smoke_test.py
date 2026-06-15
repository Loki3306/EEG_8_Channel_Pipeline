import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score

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
    
import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))
from baselines.ridge_aad import load_subject_examples

FS = 64
WINDOW_SEC = 10

def get_mapping_data():
    if Path("/kaggle/input").exists():
        env_files = list(Path("/kaggle/input").rglob("gammatone_envelopes.pkl"))
        env_file = env_files[0] if env_files else REPO_ROOT / "data" / "gammatone_envelopes.pkl"
        map_files = list(Path("/kaggle/input").rglob("audio_mapping.json"))
        map_file = map_files[0] if map_files else REPO_ROOT / "data" / "audio_mapping.json"
    else:
        env_file = REPO_ROOT / "data" / "gammatone_envelopes.pkl"
        map_file = REPO_ROOT / "data" / "audio_mapping.json"
        
    with open(env_file, 'rb') as f:
        envelopes = pickle.load(f)
    with open(map_file, 'r') as f:
        mapping = json.load(f)
        
    return mapping, envelopes

def normalize_array(arr):
    arr = arr - arr.mean(axis=0, keepdims=True)
    scale = arr.std(axis=0, keepdims=True) + 1e-12
    return arr / scale

def normalize_array_global(arr):
    arr = arr - arr.mean(axis=0, keepdims=True)
    scale = arr.std() + 1e-12
    return arr / scale

def extract_features(env_gammatone):
    gt_mean = np.mean(env_gammatone, axis=1)
    gt_var = np.var(env_gammatone, axis=1)
    
    # Cast to float32 to simulate MatchNet's precision and destroy the 1e-12 epsilon leak
    features = np.concatenate([gt_mean, gt_var]).astype(np.float32)
    return features

def evaluate_features(X, y, groups, name):
    X = np.array(X)
    y = np.array(y)
    groups = np.array(groups)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    gkf = GroupKFold(n_splits=5)
    clf = LogisticRegression(max_iter=1000)
    
    scores = cross_val_score(clf, X_scaled, y, groups=groups, cv=gkf)
    print(f"  {name}: {scores.mean()*100:.2f}% (Std: {scores.std()*100:.2f}%)")

def run_smoke_test():
    print("Loading data...")
    mapping, envelopes = get_mapping_data()
    
    subjects_to_test = ["S8", "S10", "S12"]
    
    for sub in subjects_to_test:
        mat_path = DATA_DIR / f"{sub}_data_preproc.mat"
        if not mat_path.exists():
            continue
            
        examples = load_subject_examples(mat_path)
        
        X_raw = []
        X_global = []
        X_per_band = []
        y = []
        groups = []
        
        for i, ex in enumerate(examples):
            trial_key = f"trial_{i}"
            if sub in mapping and trial_key in mapping[sub]:
                fname_a = mapping[sub][trial_key]["wavA"]["filename"]
                fname_b = mapping[sub][trial_key]["wavB"]["filename"]
                env_a_full = envelopes[fname_a] 
                env_b_full = envelopes[fname_b] 
            else:
                continue
                
            min_len = env_a_full.shape[1]
            win_samples = WINDOW_SEC * FS
            start = 0
            
            while start + win_samples <= min_len:
                end = start + win_samples
                
                # Raw
                e_a = env_a_full[:, start:end]
                e_b = env_b_full[:, start:end]
                feat_a = extract_features(e_a)
                feat_b = extract_features(e_b)
                X_raw.append(feat_a - feat_b)
                
                # Global (Old MatchNet)
                e_a_global = normalize_array_global(e_a.T).T
                e_b_global = normalize_array_global(e_b.T).T
                feat_a_global = extract_features(e_a_global)
                feat_b_global = extract_features(e_b_global)
                X_global.append(feat_a_global - feat_b_global)
                
                # Per-Band (New MatchNet)
                e_a_band = normalize_array(e_a.T).T
                e_b_band = normalize_array(e_b.T).T
                feat_a_band = extract_features(e_a_band)
                feat_b_band = extract_features(e_b_band)
                X_per_band.append(feat_a_band - feat_b_band)
                
                y.append(1 if ex.label == 1 else 0)
                groups.append(i)
                start += win_samples
                
        print(f"\n--- {sub} ---")
        evaluate_features(X_raw, y, groups, "Raw Envelopes (No Norm)")
        evaluate_features(X_global, y, groups, "Global Norm (Old MatchNet)")
        evaluate_features(X_per_band, y, groups, "Per-Band Norm (New MatchNet)")
        
        # Shuffled
        y_shuf = np.random.permutation(y)
        evaluate_features(X_per_band, y_shuf, groups, "Per-Band Norm + Shuffled Labels")

if __name__ == "__main__":
    run_smoke_test()
