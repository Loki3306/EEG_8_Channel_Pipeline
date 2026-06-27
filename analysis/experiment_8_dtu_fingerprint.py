import os
import sys
import pickle
import numpy as np
import scipy.linalg
from scipy.signal import welch
from sklearn.cross_decomposition import CCA
from tqdm import tqdm

# Add repository root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.ridge_aad import load_subject_examples, subject_files

def compute_hjorth_parameters(signal):
    """
    Computes Hjorth parameters for a single channel.
    signal: 1D numpy array
    Returns: (activity, mobility, complexity)
    """
    first_deriv = np.diff(signal)
    second_deriv = np.diff(first_deriv)

    var_zero = np.var(signal)
    var_d1 = np.var(first_deriv)
    var_d2 = np.var(second_deriv)

    activity = var_zero
    if activity == 0 or var_d1 == 0:
        return 0, 0, 0

    mobility = np.sqrt(var_d1 / activity)
    complexity = np.sqrt(var_d2 / var_d1) / mobility

    return activity, mobility, complexity

def compute_band_powers(f, psd):
    """
    Computes relative band powers from PSD.
    """
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45) # Keep gamma bounded
    }
    
    total_power = np.trapz(psd, f) + 1e-12
    powers = {}
    
    for band, (low, high) in bands.items():
        idx = np.logical_and(f >= low, f <= high)
        if not np.any(idx):
            powers[band] = 0.0
        else:
            band_power = np.trapz(psd[idx], f[idx])
            powers[band] = band_power / total_power
            
    return powers

def normalize_array(arr):
    # Same normalizer used by our models
    arr = arr - arr.mean(axis=0, keepdims=True)
    scale = arr.std(axis=0, keepdims=True) + 1e-12
    return arr / scale

def compute_dtu_fingerprint():
    print("\n" + "="*60)
    print("PHASE A: BUILDING DTU FINGERPRINT")
    print("="*60)

    # 1. Configuration
    fs_dtu = 64
    target_channels = ['Fp1', 'Fp2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8']
    dtu_indices = [0, 33, 6, 41, 14, 51, 22, 59] # BioSemi 64 to 8-ch
    
    # Storage for trial-level metrics
    fingerprint = {
        'eeg': {
            'mean': [], 'std': [], 'rms': [], 'skewness': [], 'kurtosis': [],
            'psd': [], 'psd_freqs': None,
            'delta': [], 'theta': [], 'alpha': [], 'beta': [], 'gamma': [],
            'covariance': [], 'eigenvalues': [],
            'hjorth_activity': [], 'hjorth_mobility': [], 'hjorth_complexity': []
        },
        'audio': {
            'mean': [], 'variance': []
        },
        'relational': {
            'cca_att_1': [], 'cca_unatt_1': []
        }
    }

    # 2. Iterate DTU subjects and trials
    s_files = subject_files()
    if not s_files:
        print("ERROR: DTU Dataset not found! Are you on Kaggle with DATA_preproc available?")
        return
        
    print(f"Found {len(s_files)} DTU subjects. Extracting features...")
    
    for s_file in tqdm(s_files, desc="Subjects"):
        examples = load_subject_examples(s_file)
        
        for ex in examples:
            # eeg is (64, T). We want (T, 8) normalized.
            eeg_8 = ex.eeg[dtu_indices, :].T
            eeg_norm = normalize_array(eeg_8)
            
            # audio is already envelope form in DTU MAT file? 
            # In DTU, wav_a and wav_b are actually the broad-band audio envelopes or gammatone? 
            # DTU wav_a shape in ridge_aad.py is (T,) i.e., 1D broad-band envelope.
            # But MatchNet expects 28-band? Wait. DTU `_data_preproc` only has 1D envelopes in `wavA` `wavB`. 
            # For this fingerprint, we will use the 1D envelope provided in the MAT file for Audio features.
            
            env_a = ex.wav_a.reshape(-1, 1)
            env_b = ex.wav_b.reshape(-1, 1)
            
            env_att = env_a if ex.label == 1 else env_b
            env_unatt = env_b if ex.label == 1 else env_a
            
            env_att_norm = normalize_array(env_att)
            env_unatt_norm = normalize_array(env_unatt)
            
            min_len = min(len(eeg_norm), len(env_att_norm))
            eeg_norm = eeg_norm[:min_len]
            env_att_norm = env_att_norm[:min_len]
            env_unatt_norm = env_unatt_norm[:min_len]

            # --- EEG Features ---
            # Statistical
            fingerprint['eeg']['mean'].append(np.mean(eeg_norm, axis=0))
            fingerprint['eeg']['std'].append(np.std(eeg_norm, axis=0))
            fingerprint['eeg']['rms'].append(np.sqrt(np.mean(eeg_norm**2, axis=0)))
            
            from scipy.stats import skew, kurtosis
            fingerprint['eeg']['skewness'].append(skew(eeg_norm, axis=0))
            fingerprint['eeg']['kurtosis'].append(kurtosis(eeg_norm, axis=0))
            
            # Spectral
            psds = []
            band_powers = {'delta': [], 'theta': [], 'alpha': [], 'beta': [], 'gamma': []}
            for ch in range(8):
                f, p = welch(eeg_norm[:, ch], fs=fs_dtu, nperseg=fs_dtu*2) # 2-sec window
                psds.append(p)
                bp = compute_band_powers(f, p)
                for b in band_powers: band_powers[b].append(bp[b])
                if fingerprint['eeg']['psd_freqs'] is None:
                    fingerprint['eeg']['psd_freqs'] = f
            
            fingerprint['eeg']['psd'].append(np.array(psds))
            fingerprint['eeg']['delta'].append(band_powers['delta'])
            fingerprint['eeg']['theta'].append(band_powers['theta'])
            fingerprint['eeg']['alpha'].append(band_powers['alpha'])
            fingerprint['eeg']['beta'].append(band_powers['beta'])
            fingerprint['eeg']['gamma'].append(band_powers['gamma'])
            
            # Spatial
            cov = np.cov(eeg_norm, rowvar=False)
            fingerprint['eeg']['covariance'].append(cov)
            eigenvals = scipy.linalg.eigh(cov, eigvals_only=True)
            fingerprint['eeg']['eigenvalues'].append(np.sort(eigenvals)[::-1]) # descending
            
            # Temporal (Hjorth)
            act, mob, comp = [], [], []
            for ch in range(8):
                a, m, c = compute_hjorth_parameters(eeg_norm[:, ch])
                act.append(a); mob.append(m); comp.append(c)
            fingerprint['eeg']['hjorth_activity'].append(act)
            fingerprint['eeg']['hjorth_mobility'].append(mob)
            fingerprint['eeg']['hjorth_complexity'].append(comp)
            
            # --- Audio Features ---
            fingerprint['audio']['mean'].append(np.mean(env_att_norm))
            fingerprint['audio']['variance'].append(np.var(env_att_norm))
            
            # --- Relational Features ---
            cca = CCA(n_components=1)
            try:
                cca.fit(eeg_norm, env_att_norm)
                x_c, y_c = cca.transform(eeg_norm, env_att_norm)
                r_att = np.corrcoef(x_c[:, 0], y_c[:, 0])[0, 1]
            except Exception:
                r_att = 0.0
                
            try:
                cca.fit(eeg_norm, env_unatt_norm)
                x_c, y_c = cca.transform(eeg_norm, env_unatt_norm)
                r_unatt = np.corrcoef(x_c[:, 0], y_c[:, 0])[0, 1]
            except Exception:
                r_unatt = 0.0
                
            fingerprint['relational']['cca_att_1'].append(r_att)
            fingerprint['relational']['cca_unatt_1'].append(r_unatt)

    # 3. Aggregate
    print("Aggregating metrics...")
    dtu_profile = {}
    
    # Helper to average lists of arrays
    def aggregate_dict(d):
        agg = {}
        for k, v in d.items():
            if k == 'psd_freqs':
                agg[k] = v
                continue
            if isinstance(v, list) and len(v) > 0:
                agg[k] = np.mean(np.array(v), axis=0)
            elif isinstance(v, dict):
                agg[k] = aggregate_dict(v)
        return agg

    dtu_profile['eeg'] = aggregate_dict(fingerprint['eeg'])
    dtu_profile['audio'] = aggregate_dict(fingerprint['audio'])
    dtu_profile['relational'] = aggregate_dict(fingerprint['relational'])
    
    # 4. Save
    os.makedirs('data', exist_ok=True)
    out_path = 'data/DTU_Profile.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(dtu_profile, f)
        
    print(f"\n[+] Successfully saved DTU Fingerprint to {out_path}")
    print("\nSummary Statistics:")
    print(f"Mean Canonical Correlation (Attended):   {dtu_profile['relational']['cca_att_1']:.4f}")
    print(f"Mean Canonical Correlation (Unattended): {dtu_profile['relational']['cca_unatt_1']:.4f}")

if __name__ == "__main__":
    compute_dtu_fingerprint()
