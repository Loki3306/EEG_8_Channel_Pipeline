import os
import sys
import re
import math
import numpy as np
import scipy.io
import scipy.signal
import torch
from pathlib import Path
import argparse

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.extract_gammatone_envelopes import extract_gammatone_envelopes

FS = 64

def get_kul_subject_files(dataset_dir=None):
    """Finds and sorts KUL subject files (S1 to S16)."""
    base_dirs = []
    if dataset_dir:
        base_dirs.append(Path(dataset_dir))
    base_dirs.extend([
        Path("/kaggle/input/datasets/lowk1ee/s1-klu/"),
        Path("/kaggle/input/s1-klu/"),
        REPO_ROOT / "data" / "s1-klu",
        Path("data")
    ])
    
    files = []
    for d in base_dirs:
        if d.exists():
            files = list(d.rglob("S*.mat"))
            if files:
                break
                
    if not files:
        print("Warning: Could not find KUL dataset directory.")
        return []
        
    subj_files = []
    for f in files:
        m = re.search(r"S(\d+)", f.name, re.IGNORECASE)
        if m:
            subj_files.append((int(m.group(1)), f))
            
    subj_files.sort(key=lambda x: x[0])
    return [f for idx, f in subj_files]

def load_kul_trials(mat_path):
    mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    trials = mat['trials'] if 'trials' in mat else mat['trial']
    if not isinstance(trials, np.ndarray):
        trials = [trials]
    return trials

def preprocess_trial(trial, envelope_cache, apply_car=True):
    try:
        eeg_data = trial.RawData.EegData
        fs_eeg = trial.FileHeader.SampleRate
        channel_names = [ch.Label for ch in trial.FileHeader.Channels]
    except AttributeError:
        return None, None, None, "Invalid EEG shape or missing metadata"
        
    target_channels = ['T7', 'C2', 'FT8', 'P7', 'CPz', 'Fp1', 'TP8', 'C3']
    
    if apply_car:
        eeg_data = eeg_data - eeg_data.mean(axis=1, keepdims=True)
        
    try:
        sel_idx = [channel_names.index(tc) if tc in channel_names else [c.upper() for c in channel_names].index(tc.upper()) for tc in target_channels]
    except ValueError:
        return None, None, None, f"Bad channels (Expected: {target_channels})"
        
    eeg_8 = eeg_data[:, sel_idx]
    
    nyq = 0.5 * fs_eeg
    b, a = scipy.signal.butter(4, [1.0/nyq, 8.0/nyq], btype='band')
    eeg_8 = scipy.signal.filtfilt(b, a, eeg_8, axis=0)
    
    g = math.gcd(FS, int(fs_eeg))
    eeg_8 = scipy.signal.resample_poly(eeg_8, FS // g, int(fs_eeg) // g, axis=0)
    
    arr = eeg_8 - eeg_8.mean(axis=0, keepdims=True)
    scale = arr.std(axis=0, keepdims=True) + 1e-12
    eeg_norm = arr / scale
    
    try:
        att_ear = trial.attended_ear
    except AttributeError:
        return None, None, None, "Missing attended_ear"
        
    try:
        stimuli = trial.stimuli
    except AttributeError:
        return None, None, None, "Missing stimuli"
        
    if len(stimuli) < 2: 
        return None, None, None, "Less than 2 stimuli"
        
    att_wav_name = str(stimuli[0] if att_ear == 'L' else stimuli[1]).strip()
    unatt_wav_name = str(stimuli[1] if att_ear == 'L' else stimuli[0]).strip()
    
    # Direct path construction
    if os.path.exists("/kaggle/input/datasets/lowk1ee/audio-klu/stimuli"):
        stimuli_dir = "/kaggle/input/datasets/lowk1ee/audio-klu/stimuli"
    elif os.path.exists("/kaggle/input/audio-klu/stimuli"):
        stimuli_dir = "/kaggle/input/audio-klu/stimuli"
    else:
        stimuli_dir = os.path.join(REPO_ROOT, "data", "audio-klu", "stimuli")
        
    att_wav_path = os.path.join(stimuli_dir, att_wav_name)
    unatt_wav_path = os.path.join(stimuli_dir, unatt_wav_name)
    
    if not os.path.isfile(att_wav_path):
        raise FileNotFoundError(f"Missing stimulus file: {att_wav_path}")
    if not os.path.isfile(unatt_wav_path):
        raise FileNotFoundError(f"Missing stimulus file: {unatt_wav_path}")
        
    if att_wav_path not in envelope_cache:
        envelope_cache[att_wav_path] = extract_gammatone_envelopes(att_wav_path, target_fs=FS)
    if unatt_wav_path not in envelope_cache:
        envelope_cache[unatt_wav_path] = extract_gammatone_envelopes(unatt_wav_path, target_fs=FS)
        
    env_att = envelope_cache[att_wav_path]
    env_unatt = envelope_cache[unatt_wav_path]
    
    def norm_env(env):
        env = env.T
        env = env - env.mean(axis=0, keepdims=True)
        env = env / (env.std(axis=0, keepdims=True) + 1e-12)
        return env.T
        
    env_att = norm_env(env_att)
    env_unatt = norm_env(env_unatt)
    
    min_len = min(len(eeg_norm), env_att.shape[1], env_unatt.shape[1])
    if min_len < FS * 5:
        return None, None, None, "Too-short recording"
        
    return eeg_norm[:min_len].T, env_att[:, :min_len], env_unatt[:, :min_len], "Success"

def build_cache(dataset_dir=None, output_dir="data/processed_kul"):
    out_path = REPO_ROOT / output_dir
    out_path.mkdir(parents=True, exist_ok=True)
    
    subject_files = get_kul_subject_files(dataset_dir)
    if not subject_files:
        print("No KUL subject files found to process.")
        return
        
    print(f"Found {len(subject_files)} KUL subjects. Building offline cache in {out_path} ...")
    
    computed_envelope_cache = {}
    
    for filepath in sorted(subject_files):
        sub_id = re.search(r'(S\d+)', filepath.name, re.IGNORECASE).group(1).upper()
        save_file = out_path / f"{sub_id}.pt"
        
        if save_file.exists():
            print(f"[{sub_id}] Cache already exists at {save_file}, skipping.")
            continue
            
        trials = load_kul_trials(str(filepath))
        valid_trials = []
        discard_reasons = {}
        
        print(f"\nProcessing Subject {sub_id} ({len(trials)} trials)...")
        for i, t in enumerate(trials):
            sys.stdout.write(f"\r  Trial {i+1}/{len(trials)}")
            sys.stdout.flush()
            
            x, ya, yb, reason = preprocess_trial(t, computed_envelope_cache, apply_car=True)
            if x is not None:
                meta = {
                    "TrialID": getattr(t, "TrialID", i+1),
                    "experiment": getattr(t, "experiment", "Unknown"),
                    "attended_ear": att_ear,
                    "attended_track": "1" if att_ear == 'L' else "2"
                }
                
                # Convert to torch tensors to save loading time during training
                valid_trials.append({
                    "meta": meta,
                    "eeg": torch.FloatTensor(x),
                    "audio_a": torch.FloatTensor(ya),
                    "audio_b": torch.FloatTensor(yb)
                })
            else:
                discard_reasons[reason] = discard_reasons.get(reason, 0) + 1
                
        print(f"\n[{sub_id}] Processed {len(valid_trials)} valid trials. Discarded {len(trials) - len(valid_trials)}.")
        if discard_reasons:
            for r, c in discard_reasons.items():
                print(f"  - {r}: {c}")
                
        # Save to disk
        data_dict = {
            "subject_id": sub_id,
            "trials": valid_trials
        }
        torch.save(data_dict, save_file)
        print(f"[{sub_id}] Saved cached data to {save_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default=None, help="Path to KUL dataset")
    parser.add_argument("--output_dir", type=str, default="data/processed_kul", help="Directory to save .pt files")
    args = parser.parse_args()
    
    build_cache(args.dataset_dir, args.output_dir)
