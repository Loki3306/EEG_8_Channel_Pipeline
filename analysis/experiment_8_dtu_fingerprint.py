import os
import sys
import pickle
import numpy as np
import scipy.linalg
from scipy.signal import welch
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add repository root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.ridge_aad import load_subject_examples, subject_files
try:
    from models.matchnet import ContrastiveMatchNet
except ImportError:
    ContrastiveMatchNet = None

def compute_hjorth_parameters(signal):
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
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30)
    }
    # np.trapezoid for numpy 2.0, fallback to np.trapz
    try:
        from numpy import trapezoid as trapz
    except ImportError:
        from numpy import trapz
        
    total_power = trapz(psd, f) + 1e-12
    powers = {}
    for band, (low, high) in bands.items():
        idx = np.logical_and(f >= low, f <= high)
        if not np.any(idx):
            powers[band] = 0.0
        else:
            band_power = trapz(psd[idx], f[idx])
            powers[band] = band_power / total_power
    return powers

def normalize_array(arr):
    arr = arr - arr.mean(axis=0, keepdims=True)
    scale = arr.std(axis=0, keepdims=True) + 1e-12
    return arr / scale

def get_dtu_mapping_and_envelopes():
    from pathlib import Path
    import json
    REPO_ROOT = Path(__file__).resolve().parents[1]
    
    kaggle_map_dir = Path("/kaggle/input/datasets/lokeshgile/dataset-eeg")
    if (kaggle_map_dir / "audio_mapping.json").exists():
        map_file = kaggle_map_dir / "audio_mapping.json"
    else:
        map_file = REPO_ROOT / "data" / "audio_mapping.json"
    
    kaggle_env_dir = Path("/kaggle/input/datasets/lokeshgile/gammatone-envelope")
    if kaggle_env_dir.exists() and list(kaggle_env_dir.glob("*.pkl")):
        env_file = list(kaggle_env_dir.glob("*.pkl"))[0]
    else:
        env_file = REPO_ROOT / "data" / "gammatone_envelopes.pkl"
        
    if not map_file.exists() or not env_file.exists():
        return None, None
        
    with open(map_file, 'r') as f:
        mapping = json.load(f)
    with open(env_file, 'rb') as f:
        envelopes = pickle.load(f)
    return mapping, envelopes

def load_matchnet(device='cuda'):
    if ContrastiveMatchNet is None:
        return None
    model = ContrastiveMatchNet(
        eeg_model_type="eegnet",
        eeg_channels=8,
        audio_channels=28,
        latent_dim=64
    )
    checkpoint_path = "/kaggle/working/EEG_8_Channel_Pipeline/checkpoints/matchnet_fold_S2_data_preproc_best.pth"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Loaded MatchNet checkpoint successfully.")
    else:
        print("MatchNet checkpoint NOT found. Ensure you are on Kaggle and path is correct.")
        return None
    model.to(device)
    model.eval()
    return model

def compute_dtu_fingerprint():
    print("\n" + "="*60)
    print("PHASE A: BUILDING DTU FINGERPRINT")
    print("="*60)

    # 1. Configuration
    fs_dtu = 64
    dtu_indices = [13, 46, 43, 23, 50, 0, 52, 14] # Correct DTU training channels
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_matchnet(device)
    
    # Storage for trial-level metrics
    fingerprint = {
        'metadata': {
            'num_subjects': 0, 'num_trials': 0, 'fs': fs_dtu,
            'channels': dtu_indices, 'normalization': 'z-score per trial'
        },
        'eeg': {
            'raw_mean': [], 'raw_std': [], 'raw_rms': [], 'raw_skewness': [], 'raw_kurtosis': [],
            'psd': [], 'psd_freqs': None, 'psd_peak_freq': [],
            'delta': [], 'theta': [], 'alpha': [], 'beta': [],
            'covariance': [], 'correlation': [], 'cov_cond_num': [], 'eigenvalues': [],
            'hjorth_activity': [], 'hjorth_mobility': [], 'hjorth_complexity': []
        },
        'matchnet': {
            'embeddings': [], 'embedding_cov': [], 'embedding_norm': [],
            'cosine_sim_a': [], 'cosine_sim_b': [], 'margin': []
        }
    }

    s_files = subject_files()
    if not s_files:
        print("ERROR: DTU Dataset not found! Are you on Kaggle with DATA_preproc available?")
        return
        
    print(f"Found {len(s_files)} DTU subjects. Extracting features...")
    
    from data.extract_gammatone_envelopes import extract_gammatone_envelopes
    
    # We will need mapping.json to find 28-band envelopes if we want latent space evaluation
    mapping, envelopes = get_dtu_mapping_and_envelopes()
    if mapping is None or envelopes is None:
        print("WARNING: 28-band gammatone envelopes not found. MatchNet evaluation will be skipped.")

    for s_file in tqdm(s_files, desc="Subjects"):
        subj = s_file.stem.split('_')[0]
        examples = load_subject_examples(s_file)
        fingerprint['metadata']['num_subjects'] += 1
        
        for ex in examples:
            fingerprint['metadata']['num_trials'] += 1
            
            # eeg is (64, T). We want (T, 8)
            eeg_8_raw = ex.eeg[dtu_indices, :].T
            eeg_norm = normalize_array(eeg_8_raw)

            # --- EEG Features (Raw) ---
            fingerprint['eeg']['raw_mean'].append(np.mean(eeg_8_raw, axis=0))
            fingerprint['eeg']['raw_std'].append(np.std(eeg_8_raw, axis=0))
            fingerprint['eeg']['raw_rms'].append(np.sqrt(np.mean(eeg_8_raw**2, axis=0)))
            
            from scipy.stats import skew, kurtosis
            fingerprint['eeg']['raw_skewness'].append(skew(eeg_8_raw, axis=0))
            fingerprint['eeg']['raw_kurtosis'].append(kurtosis(eeg_8_raw, axis=0))
            
            # --- EEG Features (Normalized) ---
            # Spectral
            psds = []
            band_powers = {'delta': [], 'theta': [], 'alpha': [], 'beta': []}
            peak_freqs = []
            for ch in range(8):
                f, p = welch(eeg_norm[:, ch], fs=fs_dtu, nperseg=fs_dtu*4) # 4-sec window (0.25 Hz res)
                psds.append(p)
                bp = compute_band_powers(f, p)
                for b in band_powers: band_powers[b].append(bp[b])
                
                # Peak frequency in the 1-30Hz range
                valid_idx = np.logical_and(f >= 1, f <= 30)
                peak_freqs.append(f[valid_idx][np.argmax(p[valid_idx])])
                
                if fingerprint['eeg']['psd_freqs'] is None:
                    fingerprint['eeg']['psd_freqs'] = f
            
            fingerprint['eeg']['psd'].append(np.array(psds))
            fingerprint['eeg']['psd_peak_freq'].append(peak_freqs)
            fingerprint['eeg']['delta'].append(band_powers['delta'])
            fingerprint['eeg']['theta'].append(band_powers['theta'])
            fingerprint['eeg']['alpha'].append(band_powers['alpha'])
            fingerprint['eeg']['beta'].append(band_powers['beta'])
            
            # Spatial
            cov = np.cov(eeg_norm, rowvar=False)
            corr = np.corrcoef(eeg_norm, rowvar=False)
            fingerprint['eeg']['covariance'].append(cov)
            fingerprint['eeg']['correlation'].append(corr)
            
            cond_num = np.linalg.cond(cov)
            fingerprint['eeg']['cov_cond_num'].append(cond_num)
            
            eigenvals = scipy.linalg.eigh(cov, eigvals_only=True)
            fingerprint['eeg']['eigenvalues'].append(np.sort(eigenvals)[::-1])
            
            # Temporal (Hjorth)
            act, mob, comp = [], [], []
            for ch in range(8):
                a, m, c = compute_hjorth_parameters(eeg_norm[:, ch])
                act.append(a); mob.append(m); comp.append(c)
            fingerprint['eeg']['hjorth_activity'].append(act)
            fingerprint['eeg']['hjorth_mobility'].append(mob)
            fingerprint['eeg']['hjorth_complexity'].append(comp)
            
            # --- MatchNet Evaluation ---
            if model is not None and mapping is not None and envelopes:
                trial_key = f"trial{ex.trial_index+1}"
                if subj in mapping and trial_key in mapping[subj]:
                    fname_a = mapping[subj][trial_key]["wavA"]["filename"]
                    fname_b = mapping[subj][trial_key]["wavB"]["filename"]
                    
                    if fname_a in envelopes and fname_b in envelopes:
                        env_a = envelopes[fname_a]
                        env_b = envelopes[fname_b]
                        
                        env_att = env_a if ex.label == 1 else env_b
                        env_unatt = env_b if ex.label == 1 else env_a
                        
                        env_att = normalize_array(env_att.T).T
                        env_unatt = normalize_array(env_unatt.T).T
                        
                        min_len = min(len(eeg_norm), env_att.shape[1], env_unatt.shape[1])
                        
                        win_len = 3 * fs_dtu
                        stride = 96 # 1.5 seconds at 64Hz
                        
                        all_eeg_emb = []
                        all_att_emb = []
                        all_unatt_emb = []
                        
                        # Generate windows
                        for start in range(0, min_len - win_len + 1, stride):
                            eeg_win = eeg_norm[start:start+win_len]
                            att_win = env_att[:, start:start+win_len]
                            unatt_win = env_unatt[:, start:start+win_len]
                            
                            e_tensor = torch.tensor(eeg_win.T, dtype=torch.float32).unsqueeze(0).to(device)
                            a_tensor = torch.tensor(att_win, dtype=torch.float32).unsqueeze(0).to(device)
                            u_tensor = torch.tensor(unatt_win, dtype=torch.float32).unsqueeze(0).to(device)
                            
                            with torch.no_grad():
                                e_emb = model.eeg_encoder(e_tensor)
                                e_emb = F.normalize(e_emb, p=2, dim=1)
                                
                                a_emb = model.audio_encoder(a_tensor)
                                a_emb = F.normalize(a_emb, p=2, dim=1)
                                
                                u_emb = model.audio_encoder(u_tensor)
                                u_emb = F.normalize(u_emb, p=2, dim=1)
                                
                                all_eeg_emb.append(e_emb.cpu().numpy()[0])
                                all_att_emb.append(a_emb.cpu().numpy()[0])
                                all_unatt_emb.append(u_emb.cpu().numpy()[0])
                                
                        if all_eeg_emb:
                            all_eeg_emb = np.array(all_eeg_emb)
                            all_att_emb = np.array(all_att_emb)
                            all_unatt_emb = np.array(all_unatt_emb)
                            
                            fingerprint['matchnet']['embeddings'].append(np.mean(all_eeg_emb, axis=0))
                            fingerprint['matchnet']['embedding_cov'].append(np.cov(all_eeg_emb, rowvar=False))
                            fingerprint['matchnet']['embedding_norm'].append(np.linalg.norm(all_eeg_emb, axis=1).mean())
                            
                            sim_a = np.sum(all_eeg_emb * all_att_emb, axis=1)
                            sim_b = np.sum(all_eeg_emb * all_unatt_emb, axis=1)
                            
                            fingerprint['matchnet']['cosine_sim_a'].extend(sim_a)
                            fingerprint['matchnet']['cosine_sim_b'].extend(sim_b)
                            fingerprint['matchnet']['margin'].extend(sim_a - sim_b)

    # 3. Aggregate
    print("Aggregating metrics...")
    dtu_profile = {}
    
    # Helper to preserve the full distribution of trials
    def aggregate_dict(d):
        agg = {}
        for k, v in d.items():
            if k in ['psd_freqs', 'channels', 'normalization', 'num_subjects', 'num_trials', 'fs']:
                agg[k] = v
                continue
            if isinstance(v, list) and len(v) > 0:
                # v is a list of arrays from each trial
                # Stack them to shape (N_trials, ...) to preserve the distribution
                try:
                    agg[k] = np.stack(v, axis=0)
                except ValueError:
                    # In case of jagged arrays (which shouldn't happen), fallback to object array
                    agg[k] = np.array(v, dtype=object)
            elif isinstance(v, dict):
                agg[k] = aggregate_dict(v)
            else:
                agg[k] = v
        return agg

    dtu_profile['metadata'] = aggregate_dict(fingerprint['metadata'])
    dtu_profile['eeg'] = aggregate_dict(fingerprint['eeg'])
    dtu_profile['matchnet'] = aggregate_dict(fingerprint['matchnet'])
    
    # 4. Save
    os.makedirs('data', exist_ok=True)
    out_path = 'data/DTU_Profile.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(dtu_profile, f)
        
    print(f"\n[+] Successfully saved DTU Fingerprint to {out_path}")
    print("\nSummary Statistics:")
    print(f"Num Subjects: {dtu_profile['metadata'].get('num_subjects', 0)}")
    print(f"Num Trials:   {dtu_profile['metadata'].get('num_trials', 0)}")
    if dtu_profile['matchnet'].get('margin') is not None and len(dtu_profile['matchnet']['margin']) > 0:
        print(f"Mean Margin:  {dtu_profile['matchnet']['margin'].mean():.4f}")

if __name__ == "__main__":
    compute_dtu_fingerprint()
