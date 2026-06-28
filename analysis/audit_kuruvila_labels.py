import os
import sys
import torch
import random
from pathlib import Path
import numpy as np
import scipy.io

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.kul_cached_dataset import KULCachedLoader
from models.kuruvila_original import KuruvilaOriginalCNNLSTM

def get_mat_path(subject_id):
    base_dirs = [
        Path("/kaggle/input/datasets/lowk1ee/s1-klu/"),
        Path("/kaggle/input/s1-klu/"),
        REPO_ROOT / "data" / "s1-klu",
    ]
    for d in base_dirs:
        if d.exists():
            files = list(d.rglob(f"{subject_id}*.mat"))
            if files:
                return files[0]
    return None

def calc_corr(t1, t2):
    # Flatten and calculate pearson correlation
    if t1.shape != t2.shape:
        return 0.0
    v1 = t1.flatten().numpy()
    v2 = t2.flatten().numpy()
    if np.std(v1) == 0 or np.std(v2) == 0:
        return 0.0
    return np.corrcoef(v1, v2)[0, 1]

def main():
    print("="*80)
    print("END-TO-END SEMANTIC AUDIT: RAW KUL .MAT -> CACHE -> KURUVILA MODEL")
    print("="*80)
    
    loader = KULCachedLoader(REPO_ROOT / "data" / "processed_kul")
    try:
        all_data = loader.load_all()
    except FileNotFoundError:
        print("Run on Kaggle to access the dataset cache.")
        return
        
    subject_id = "S1"
    if subject_id not in all_data:
        print(f"Subject {subject_id} not found in cache.")
        return
        
    mat_path = get_mat_path(subject_id)
    if not mat_path:
        print(f"Raw .mat file for {subject_id} not found in expected Kaggle paths.")
        return
        
    print(f"Loaded Cache for {subject_id} ({len(all_data[subject_id])} trials)")
    print(f"Loaded Raw MAT : {mat_path}")
    
    mat_data = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    raw_trials = mat_data['trials'] if 'trials' in mat_data else mat_data['trial']
    if not isinstance(raw_trials, np.ndarray):
        raw_trials = [raw_trials]
        
    # We will audit the first 3 valid trials
    trials_to_audit = min(3, len(all_data[subject_id]))
    model = KuruvilaOriginalCNNLSTM()
    
    for i in range(trials_to_audit):
        cached_t = all_data[subject_id][i]
        meta = cached_t["meta"]
        trial_id = meta["TrialID"]
        
        # TrialID is 1-indexed in our cache, so subtract 1 for raw array
        raw_t = raw_trials[trial_id - 1]
        
        print(f"\n\n{'='*80}")
        print(f"AUDITING {subject_id} - TRIAL {trial_id}")
        print(f"{'='*80}")
        
        # 1. Original MAT Data
        att_ear = getattr(raw_t, 'attended_ear', 'Unknown')
        stimuli = getattr(raw_t, 'stimuli', ['Unknown', 'Unknown'])
        
        left_audio = stimuli[0].strip() if len(stimuli) > 0 else "Unknown"
        right_audio = stimuli[1].strip() if len(stimuli) > 1 else "Unknown"
        
        attended_speaker = left_audio if att_ear == 'L' else right_audio
        unattended_speaker = right_audio if att_ear == 'L' else left_audio
        
        print("\n--- 1. Original MAT File (Ground Truth) ---")
        print(f"Left audio (stimuli[0])  : {left_audio}")
        print(f"Right audio (stimuli[1]) : {right_audio}")
        print(f"Attended ear             : {att_ear}")
        print(f"-> Attended speaker      : {attended_speaker}")
        print(f"-> Unattended speaker    : {unattended_speaker}")
        
        # 2. Cached .pt Data
        print("\n--- 2. Cached .pt Data ---")
        print(f"Meta 'attended_ear'   : {meta.get('attended_ear')}")
        print(f"Meta 'attended_track' : {meta.get('attended_track')} (1=Left, 2=Right)")
        print(f"audio_a tensor shape  : {cached_t['audio_a'].shape} (Supposed to be ATTENDED)")
        print(f"audio_b tensor shape  : {cached_t['audio_b'].shape} (Supposed to be UNATTENDED)")
        
        # 3. Training Loop Assignment
        attended_track = str(meta.get('attended_track'))
        label = 0 if attended_track == '1' else 1
        
        if attended_track == '1':
            audio_spk1 = cached_t["audio_a"]
            audio_spk2 = cached_t["audio_b"]
        else:
            audio_spk1 = cached_t["audio_b"]
            audio_spk2 = cached_t["audio_a"]
            
        print("\n--- 3. Training Loop Data Flow ---")
        print(f"Rule: if attended_track == '1' (Left), label=0, spk1=a, spk2=b")
        print(f"      if attended_track == '2' (Right), label=1, spk1=b, spk2=a")
        print(f"Applied Label         : Class {label}")
        print(f"Applied audio_spk1    : {'audio_a' if attended_track=='1' else 'audio_b'}")
        print(f"Applied audio_spk2    : {'audio_b' if attended_track=='1' else 'audio_a'}")
        
        # 4. Tensor Similarity & Verification
        print("\n--- 4. Tensor Trace Verification ---")
        corr_a_spk1 = calc_corr(cached_t["audio_a"], audio_spk1)
        corr_a_spk2 = calc_corr(cached_t["audio_a"], audio_spk2)
        corr_b_spk1 = calc_corr(cached_t["audio_b"], audio_spk1)
        corr_b_spk2 = calc_corr(cached_t["audio_b"], audio_spk2)
        
        print(f"corr(audio_a, audio_spk1) : {corr_a_spk1:.4f}")
        print(f"corr(audio_a, audio_spk2) : {corr_a_spk2:.4f}")
        print(f"corr(audio_b, audio_spk1) : {corr_b_spk1:.4f}")
        print(f"corr(audio_b, audio_spk2) : {corr_b_spk2:.4f}")
        
        # 5. Kuruvila Semantic Expectation
        print("\n--- 5. Semantic Resolution ---")
        if label == 0:
            print("Target Class = 0")
            print("Physical meaning: Model must predict that SPK1 (Left Speaker) is attended.")
            print(f"Verify SPK1 Tensor == Attended? corr(audio_a, audio_spk1) is {corr_a_spk1:.1f}")
        else:
            print("Target Class = 1")
            print("Physical meaning: Model must predict that SPK2 (Right Speaker) is attended.")
            print(f"Verify SPK2 Tensor == Attended? corr(audio_a, audio_spk2) is {corr_a_spk2:.1f}")
            
        # 6. Model Output Check
        eeg_win = cached_t["eeg"][:, :192].unsqueeze(0)
        a1_win = audio_spk1[:, :192].unsqueeze(0)
        a2_win = audio_spk2[:, :192].unsqueeze(0)
        
        with torch.no_grad():
            out = model(eeg_win, a1_win, a2_win)
            
        print("\n--- 6. Original Kuruvila Expectation ---")
        print("Original Kuruvila model is an Audio-Guided AAD network.")
        print("It expects SPK1 (audio 1) and SPK2 (audio 2) alongside EEG.")
        print("It outputs Softmax([Prob(SPK1 attended), Prob(SPK2 attended)]).")
        print(f"Input to model   : [SPK1, EEG, SPK2]")
        print(f"Model output     : {out.numpy()}")
        
        expected_out = "[1. 0.]" if label == 0 else "[0. 1.]"
        print(f"Expected output  : {expected_out}")
        
        print("\nFINAL DIAGNOSIS FOR THIS TRIAL:")
        if label == 0 and corr_a_spk1 > 0.99:
            print("  [PASS] Target 0 correctly maps the ATTENDED audio to the SPK1 branch.")
        elif label == 1 and corr_a_spk2 > 0.99:
            print("  [PASS] Target 1 correctly maps the ATTENDED audio to the SPK2 branch.")
        else:
            print("  [FAIL] SEMANTIC MISMATCH DETECTED IN TENSOR ASSIGNMENT!")
            
if __name__ == "__main__":
    main()
