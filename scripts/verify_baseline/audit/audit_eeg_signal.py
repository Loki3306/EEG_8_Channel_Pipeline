"""
Diagnostic: Ground-Truth Audit of DTU EEG-Speech Pairing & Attention Labels
=============================================================================
Answers decisively:
1. Are wavA/wavB interpreted correctly as attended/unattended?
2. Does ex.label (1 vs 2) indicate attention or speaker gender?
3. Which pairing rule gives positive cross-correlation (Delta r > 0) at physiological lags?

Usage (Kaggle):
    !cd /kaggle/working/ISEF_Project && python scripts/verify_baseline/audit/audit_eeg_signal.py
"""

from __future__ import annotations
import sys, json, pickle, os
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
from scipy.signal import butter, filtfilt

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "verify_baseline"))

from baselines.ridge_aad import subject_files, load_subject_examples

FS = 64
PHYSIO_LAGS_MS = [0, 50, 100, 150, 200, 250, 300]
TEST_SUBJECTS = ["S1_data_preproc", "S2_data_preproc"]

def butter_bandpass(x: np.ndarray, lowcut: float, highcut: float, fs: int = FS, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, x, axis=0)

def normalize_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    x = x - np.mean(x)
    return x / (np.std(x) + 1e-12)

def pearson_lag(eeg_sig: np.ndarray, audio_sig: np.ndarray, lag_ms: float) -> float:
    """Computes correlation when EEG lags audio by lag_ms (audio leads EEG)."""
    lag_samples = int(round(lag_ms * FS / 1000.0))
    n = min(len(eeg_sig), len(audio_sig))
    if lag_samples >= 0:
        e = eeg_sig[lag_samples:n]
        a = audio_sig[:n - lag_samples]
    else:
        abs_lag = abs(lag_samples)
        e = eeg_sig[:n - abs_lag]
        a = audio_sig[abs_lag:n]
    if len(e) < 64:
        return 0.0
    r, _ = pearsonr(e - np.mean(e), a - np.mean(a))
    return float(r)

def find_mapping_and_envelopes():
    base_dir = Path("/kaggle/input")
    # 1. Mapping file
    map_files = list(base_dir.rglob("audio_mapping.json"))
    map_file = map_files[0] if map_files else (REPO_ROOT / "scripts" / "verify_baseline" / "data" / "audio_mapping.json")
    
    # 2. Envelopes
    pkl_files = list(base_dir.rglob("*gammatone*.pkl"))
    if not pkl_files:
        pkl_files = list(base_dir.rglob("*.pkl"))
    env_file = pkl_files[0] if pkl_files else None
    
    mapping = None
    if map_file.exists():
        with open(map_file, "r") as f:
            mapping = json.load(f)
            
    envelopes = None
    if env_file and env_file.exists():
        with open(env_file, "rb") as f:
            envelopes = pickle.load(f)
            
    return mapping, envelopes, map_file, env_file

def run():
    print("\n" + "="*85)
    print(" GROUND-TRUTH AUDIT: DTU EEG-SPEECH PAIRING & LABEL SEMANTICS")
    print("="*85)
    
    paths = subject_files()
    if not paths:
        print("[!] No subject MAT files found.")
        return
        
    test_paths = [p for p in paths if p.stem in TEST_SUBJECTS] or paths[:2]
    mapping, envelopes, map_path, env_path = find_mapping_and_envelopes()
    print(f"Map File: {map_path}")
    print(f"Env File: {env_path}\n")

    # =========================================================================
    # PART 1: PRINT EXACT SAMPLE AUDIT (FIRST 20 TRIALS ACROSS S1 & S2)
    # =========================================================================
    print("="*85)
    print(" PART 1: DETAILED TRIAL-BY-TRIAL AUDIT (FIRST 20 TRIALS)")
    print("="*85)
    print(f"{'Idx':>3} | {'Subj':>4} | {'Tr':>3} | {'Label':>5} | {'wavA (Mapping)':>28} | {'wavB (Mapping)':>28}")
    print("-" * 85)
    
    sample_count = 0
    all_subject_examples = {}
    for p in test_paths:
        s_name = p.stem.split("_")[0]
        exs = load_subject_examples(p)
        all_subject_examples[s_name] = exs
        for i, ex in enumerate(exs):
            if sample_count < 20:
                tk = f"trial_{i}"
                fa = mapping[s_name][tk]["wavA"]["filename"] if (mapping and s_name in mapping and tk in mapping[s_name]) else "N/A"
                fb = mapping[s_name][tk]["wavB"]["filename"] if (mapping and s_name in mapping and tk in mapping[s_name]) else "N/A"
                print(f"{sample_count:3d} | {s_name:>4} | {i:3d} | {ex.label:5d} | {fa:>28} | {fb:>28}")
                sample_count += 1
    print("-" * 85)

    # =========================================================================
    # PART 2: 4-WAY EMPIRICAL CORRELATION TOURNAMENT
    # =========================================================================
    print("\n" + "="*85)
    print(" PART 2: EMPIRICAL AUDIT TOURNAMENT — WHICH RULE PRODUCES REAL TRACKING?")
    print(" Testing 4 Candidate Pairing Rules across all 60 trials:")
    print("   Rule A: wavA is ALWAYS attended, wavB is ALWAYS unattended")
    print("   Rule B: wavB is ALWAYS attended, wavA is ALWAYS unattended")
    print("   Rule C: label=1 -> wavA attended, label=2 -> wavB attended")
    print("   Rule D: label=1 -> wavB attended, label=2 -> wavA attended")
    print("="*85)

    # Accumulators for Delta r = r(att) - r(unatt) per lag
    results_mat = {"Rule A (wavA Always)": {l: [] for l in PHYSIO_LAGS_MS},
                   "Rule B (wavB Always)": {l: [] for l in PHYSIO_LAGS_MS},
                   "Rule C (Label 1=A, 2=B)": {l: [] for l in PHYSIO_LAGS_MS},
                   "Rule D (Label 1=B, 2=A)": {l: [] for l in PHYSIO_LAGS_MS}}
                   
    results_gam = {"Rule A (wavA Always)": {l: [] for l in PHYSIO_LAGS_MS},
                   "Rule B (wavB Always)": {l: [] for l in PHYSIO_LAGS_MS},
                   "Rule C (Label 1=A, 2=B)": {l: [] for l in PHYSIO_LAGS_MS},
                   "Rule D (Label 1=B, 2=A)": {l: [] for l in PHYSIO_LAGS_MS}}

    for s_name, exs in all_subject_examples.items():
        print(f"Auditing Subject {s_name} ({len(exs)} trials)...")
        for i, ex in enumerate(exs):
            # 1. Filter EEG to delta/theta speech tracking band [1.0, 8.0 Hz]
            eeg_raw = ex.eeg[:, :64] # 64 scalp channels
            eeg_filt = butter_bandpass(eeg_raw, 1.0, 8.0, FS)
            # Use mean of auditory/temporal responsive channels (or average across scalp)
            eeg_sig = normalize_1d(eeg_filt.mean(axis=1))
            
            # MAT envelopes (from ex.wav_a, ex.wav_b)
            ea_mat = normalize_1d(butter_bandpass(ex.wav_a.reshape(-1, 1), 1.0, 8.0, FS))
            eb_mat = normalize_1d(butter_bandpass(ex.wav_b.reshape(-1, 1), 1.0, 8.0, FS))
            
            # Gammatone envelopes (from envelopes.pkl via mapping)
            ea_gam, eb_gam = None, None
            tk = f"trial_{i}"
            if mapping and envelopes and s_name in mapping and tk in mapping[s_name]:
                fa = mapping[s_name][tk]["wavA"]["filename"]
                fb = mapping[s_name][tk]["wavB"]["filename"]
                if fa in envelopes and fb in envelopes:
                    ga = envelopes[fa]
                    gb = envelopes[fb]
                    if ga.ndim == 2: ga = ga.mean(axis=0) # broadband
                    if gb.ndim == 2: gb = gb.mean(axis=0)
                    ea_gam = normalize_1d(butter_bandpass(ga.reshape(-1, 1), 1.0, 8.0, FS))
                    eb_gam = normalize_1d(butter_bandpass(gb.reshape(-1, 1), 1.0, 8.0, FS))

            lbl = ex.label
            
            # Define Attended/Unattended streams for each rule
            # Rule A: wavA always
            mat_att_A, mat_unatt_A = ea_mat, eb_mat
            # Rule B: wavB always
            mat_att_B, mat_unatt_B = eb_mat, ea_mat
            # Rule C: label 1=A, 2=B
            mat_att_C, mat_unatt_C = (ea_mat, eb_mat) if lbl == 1 else (eb_mat, ea_mat)
            # Rule D: label 1=B, 2=A
            mat_att_D, mat_unatt_D = (eb_mat, ea_mat) if lbl == 1 else (ea_mat, eb_mat)

            for lag in PHYSIO_LAGS_MS:
                # MAT evaluations
                r_a = pearson_lag(eeg_sig, mat_att_A, lag)
                r_u = pearson_lag(eeg_sig, mat_unatt_A, lag)
                results_mat["Rule A (wavA Always)"][lag].append(r_a - r_u)
                results_mat["Rule B (wavB Always)"][lag].append(r_u - r_a)
                
                r_c_a = pearson_lag(eeg_sig, mat_att_C, lag)
                r_c_u = pearson_lag(eeg_sig, mat_unatt_C, lag)
                results_mat["Rule C (Label 1=A, 2=B)"][lag].append(r_c_a - r_c_u)
                
                r_d_a = pearson_lag(eeg_sig, mat_att_D, lag)
                r_d_u = pearson_lag(eeg_sig, mat_unatt_D, lag)
                results_mat["Rule D (Label 1=B, 2=A)"][lag].append(r_d_a - r_d_u)

                # Gammatone evaluations
                if ea_gam is not None:
                    rg_a = pearson_lag(eeg_sig, ea_gam, lag)
                    rg_u = pearson_lag(eeg_sig, eb_gam, lag)
                    results_gam["Rule A (wavA Always)"][lag].append(rg_a - rg_u)
                    results_gam["Rule B (wavB Always)"][lag].append(rg_u - rg_a)
                    
                    rg_c_a, rg_c_u = (rg_a, rg_u) if lbl == 1 else (rg_u, rg_a)
                    results_gam["Rule C (Label 1=A, 2=B)"][lag].append(rg_c_a - rg_c_u)
                    
                    rg_d_a, rg_d_u = (rg_u, rg_a) if lbl == 1 else (rg_a, rg_u)
                    results_gam["Rule D (Label 1=B, 2=A)"][lag].append(rg_d_a - rg_d_u)

    print("\n" + "="*85)
    print(" TOURNAMENT RESULTS: MEAN DELTA-R [r(Attended) - r(Unattended)] (GAMMATONE)")
    print(" True biological tracking MUST produce positive Delta-r at 100-250ms latency.")
    print("="*85)
    print(f"{'Lag (ms)':>8} | {'Rule A (wavA Always)':>20} | {'Rule B (wavB Always)':>20} | {'Rule C (1=A, 2=B)':>18} | {'Rule D (1=B, 2=A)':>18}")
    print("-" * 92)
    
    rule_scores = {r: 0.0 for r in results_gam.keys()}
    for lag in PHYSIO_LAGS_MS:
        ma = np.mean(results_gam["Rule A (wavA Always)"][lag])
        mb = np.mean(results_gam["Rule B (wavB Always)"][lag])
        mc = np.mean(results_gam["Rule C (Label 1=A, 2=B)"][lag])
        md = np.mean(results_gam["Rule D (Label 1=B, 2=A)"][lag])
        print(f"{lag:6d}ms | {ma:20.5f} | {mb:20.5f} | {mc:18.5f} | {md:18.5f}")
        if lag in [100, 150, 200]:
            rule_scores["Rule A (wavA Always)"] += ma
            rule_scores["Rule B (wavB Always)"] += mb
            rule_scores["Rule C (Label 1=A, 2=B)"] += mc
            rule_scores["Rule D (Label 1=B, 2=A)"] += md
            
    print("="*92)
    best_rule = max(rule_scores, key=rule_scores.get)
    print(f"\n[WINNING GROUND-TRUTH RULE]: {best_rule} (Cumulative Tracking Score: {rule_scores[best_rule]:+.5f})")
    print("="*92)

if __name__ == "__main__":
    run()
