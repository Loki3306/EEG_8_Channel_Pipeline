import os
import sys
import pickle
import numpy as np
import scipy.linalg
from scipy.signal import welch, butter, filtfilt, resample_poly
import scipy.io as sio
import torch
import torch.nn.functional as F
from tqdm import tqdm
import math

# Add repository root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.ridge_aad import load_subject_examples, subject_files
from data.extract_gammatone_envelopes import extract_gammatone_envelopes
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
    bands = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30)}
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

def load_matchnet(device='cuda'):
    if ContrastiveMatchNet is None: return None
    model = ContrastiveMatchNet(eeg_model_type="eegnet", eeg_channels=8, audio_channels=28, latent_dim=64)
    chk_path = "/kaggle/working/EEG_8_Channel_Pipeline/checkpoints/matchnet_fold_S2_data_preproc_best.pth"
    if os.path.exists(chk_path):
        model.load_state_dict(torch.load(chk_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def compute_kul_fingerprint():
    print("\n" + "="*60)
    print("PHASE B: KUL FINGERPRINT & DTU COMPARISON")
    print("="*60)
    
    fs_target = 64
    target_channels = ['Fp1', 'Fp2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8']
    
    mat_path = "/kaggle/input/datasets/lowk1ee/s1-klu/S1_KLU.mat"
    wav_dir = "/kaggle/input/datasets/lowk1ee/audio-klu"
    
    if not os.path.exists(mat_path):
        print(f"ERROR: KUL S1 MAT file not found at {mat_path}. Are you on Kaggle?")
        return
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_matchnet(device)
    
    print("Loading KUL S1 MAT file...")
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    trials = mat['trials'] if 'trials' in mat else mat['trial']
    
    fingerprint = {
        'eeg': {
            'raw_mean': [], 'raw_std': [], 'raw_rms': [], 'raw_skewness': [], 'raw_kurtosis': [],
            'psd': [], 'psd_peak_freq': [],
            'delta': [], 'theta': [], 'alpha': [], 'beta': [],
            'covariance': [], 'correlation': [], 'cov_cond_num': [], 'eigenvalues': [],
            'hjorth_activity': [], 'hjorth_mobility': [], 'hjorth_complexity': []
        },
        'matchnet': {
            'embeddings': [], 'embedding_cov': [], 'embedding_norm': [],
            'cosine_sim_a': [], 'cosine_sim_b': [], 'margin': []
        }
    }
    
    def find_wav(name):
        for r, d, f in os.walk(wav_dir):
            if name in f: return os.path.join(r, name)
            if name+".wav" in f: return os.path.join(r, name+".wav")
        return None
        
    # Memoize envelopes to save time
    audio_cache = {}

    print(f"Extracting features from {len(trials)} KUL trials...")
    for t_idx, trial in enumerate(tqdm(trials, desc="KUL Trials")):
        eeg_data = trial.RawData.EegData
        fs_eeg = trial.FileHeader.SampleRate
        channel_names = [ch.Label for ch in trial.FileHeader.Channels]
        
        try:
            sel_idx = [channel_names.index(tc) if tc in channel_names else [c.upper() for c in channel_names].index(tc.upper()) for tc in target_channels]
        except ValueError:
            continue
            
        eeg_8 = eeg_data[:, sel_idx]
        
        # Bandpass 1-8 Hz (MatchNet standard)
        nyq = 0.5 * fs_eeg
        b, a = butter(2, [1.0/nyq, 8.0/nyq], btype='band')
        eeg_8 = filtfilt(b, a, eeg_8, axis=0)
        
        # Downsample to 64 Hz
        g = math.gcd(fs_target, fs_eeg)
        up = fs_target // g
        down = fs_eeg // g
        eeg_8 = resample_poly(eeg_8, up, down, axis=0) # (T, 8)
        
        # Features (Raw)
        fingerprint['eeg']['raw_mean'].append(np.mean(eeg_8, axis=0))
        fingerprint['eeg']['raw_std'].append(np.std(eeg_8, axis=0))
        fingerprint['eeg']['raw_rms'].append(np.sqrt(np.mean(eeg_8**2, axis=0)))
        from scipy.stats import skew, kurtosis
        fingerprint['eeg']['raw_skewness'].append(skew(eeg_8, axis=0))
        fingerprint['eeg']['raw_kurtosis'].append(kurtosis(eeg_8, axis=0))
        
        # Features (Normalized)
        eeg_norm = normalize_array(eeg_8)
        
        psds = []
        band_powers = {'delta': [], 'theta': [], 'alpha': [], 'beta': []}
        peak_freqs = []
        for ch in range(8):
            f, p = welch(eeg_norm[:, ch], fs=fs_target, nperseg=fs_target*4)
            psds.append(p)
            bp = compute_band_powers(f, p)
            for b in band_powers: band_powers[b].append(bp[b])
            valid_idx = np.logical_and(f >= 1, f <= 30)
            peak_freqs.append(f[valid_idx][np.argmax(p[valid_idx])])
            
        fingerprint['eeg']['psd'].append(np.array(psds))
        fingerprint['eeg']['psd_peak_freq'].append(peak_freqs)
        for b in band_powers: fingerprint['eeg'][b].append(band_powers[b])
        
        cov = np.cov(eeg_norm, rowvar=False)
        fingerprint['eeg']['covariance'].append(cov)
        fingerprint['eeg']['correlation'].append(np.corrcoef(eeg_norm, rowvar=False))
        fingerprint['eeg']['cov_cond_num'].append(np.linalg.cond(cov))
        fingerprint['eeg']['eigenvalues'].append(np.sort(scipy.linalg.eigh(cov, eigvals_only=True))[::-1])
        
        act, mob, comp = [], [], []
        for ch in range(8):
            a, m, c = compute_hjorth_parameters(eeg_norm[:, ch])
            act.append(a); mob.append(m); comp.append(c)
        fingerprint['eeg']['hjorth_activity'].append(act)
        fingerprint['eeg']['hjorth_mobility'].append(mob)
        fingerprint['eeg']['hjorth_complexity'].append(comp)
        
        # --- Audio & MatchNet ---
        if model is not None:
            att_ear = trial.attended_ear
            stimuli = trial.stimuli
            
            att_wav_name = str(stimuli[0] if att_ear == 'L' else stimuli[1])
            unatt_wav_name = str(stimuli[1] if att_ear == 'L' else stimuli[0])
            
            att_wav_path = find_wav(att_wav_name)
            unatt_wav_path = find_wav(unatt_wav_name)
            
            if att_wav_path and unatt_wav_path:
                if att_wav_name not in audio_cache:
                    audio_cache[att_wav_name] = extract_gammatone_envelopes(att_wav_path, target_fs=fs_target)
                if unatt_wav_name not in audio_cache:
                    audio_cache[unatt_wav_name] = extract_gammatone_envelopes(unatt_wav_path, target_fs=fs_target)
                    
                env_att = audio_cache[att_wav_name] # (28, T)
                env_unatt = audio_cache[unatt_wav_name] # (28, T)
                
                env_att = normalize_array(env_att.T).T
                env_unatt = normalize_array(env_unatt.T).T
                
                min_len = min(len(eeg_norm), env_att.shape[1], env_unatt.shape[1])
                win_len = 3 * fs_target
                stride = fs_target
                
                all_eeg_emb = []
                all_att_emb = []
                all_unatt_emb = []
                
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
    def aggregate_dict(d):
        agg = {}
        for k, v in d.items():
            if isinstance(v, list) and len(v) > 0:
                if k in ['cosine_sim_a', 'cosine_sim_b', 'margin']:
                    agg[k] = {'mean': np.mean(v), 'std': np.std(v), 'median': np.median(v)}
                else:
                    agg[k] = np.mean(np.array(v), axis=0)
            elif isinstance(v, dict):
                agg[k] = aggregate_dict(v)
            else:
                agg[k] = v
        return agg

    kul_profile = {}
    kul_profile['eeg'] = aggregate_dict(fingerprint['eeg'])
    kul_profile['matchnet'] = aggregate_dict(fingerprint['matchnet'])
    
    os.makedirs('data', exist_ok=True)
    with open('data/KUL_Profile.pkl', 'wb') as f:
        pickle.dump(kul_profile, f)
        
    print("\n[+] Successfully saved KUL Fingerprint to data/KUL_Profile.pkl")
    
    # ---------------------------------------------------------
    # 4. Compare Distributions (DTU vs KUL)
    # ---------------------------------------------------------
    dtu_path = 'data/DTU_Profile.pkl'
    if not os.path.exists(dtu_path):
        print(f"Skipping comparison: {dtu_path} not found.")
        return
        
    with open(dtu_path, 'rb') as f:
        dtu_profile = pickle.load(f)
        
    print("\n" + "-"*60)
    print("DISTRIBUTION COMPARISON: DTU vs KUL")
    print("-"*60)
    
    from scipy.stats import wasserstein_distance
    
    comparisons = []
    
    # 1D Array comparisons
    metrics = [
        ('Raw RMS', 'raw_rms'), ('Raw Skewness', 'raw_skewness'), ('Raw Kurtosis', 'raw_kurtosis'),
        ('Alpha Power', 'alpha'), ('Theta Power', 'theta'), ('Beta Power', 'beta'),
        ('PSD Peak Freq', 'psd_peak_freq'), ('Eigenvalues', 'eigenvalues'),
        ('Hjorth Activity', 'hjorth_activity'), ('Hjorth Mobility', 'hjorth_mobility')
    ]
    
    for name, key in metrics:
        if key in dtu_profile['eeg'] and key in kul_profile['eeg']:
            dtu_val = np.array(dtu_profile['eeg'][key]).flatten()
            kul_val = np.array(kul_profile['eeg'][key]).flatten()
            w_dist = wasserstein_distance(dtu_val, kul_val)
            mean_diff = np.abs(np.mean(dtu_val) - np.mean(kul_val))
            comparisons.append({'Metric': name, 'Wasserstein': w_dist, 'MeanDiff': mean_diff})
            
    # Covariance Matrix Distance (Frobenius Norm of Difference)
    if 'covariance' in dtu_profile['eeg'] and 'covariance' in kul_profile['eeg']:
        dtu_cov = dtu_profile['eeg']['covariance']
        kul_cov = kul_profile['eeg']['covariance']
        frob = np.linalg.norm(dtu_cov - kul_cov, 'fro')
        comparisons.append({'Metric': 'Spatial Covariance (Frobenius)', 'Wasserstein': frob, 'MeanDiff': frob})
        
    # MatchNet Latent Distance
    if 'margin' in dtu_profile['matchnet'] and 'margin' in kul_profile['matchnet']:
        dtu_margin = dtu_profile['matchnet']['margin']['mean']
        kul_margin = kul_profile['matchnet']['margin']['mean']
        diff = np.abs(dtu_margin - kul_margin)
        comparisons.append({'Metric': 'MatchNet Margin Mean', 'Wasserstein': diff, 'MeanDiff': diff})
        
    if 'embeddings' in dtu_profile['matchnet'] and 'embeddings' in kul_profile['matchnet']:
        dtu_emb = dtu_profile['matchnet']['embeddings'].flatten()
        kul_emb = kul_profile['matchnet']['embeddings'].flatten()
        w_dist = wasserstein_distance(dtu_emb, kul_emb)
        comparisons.append({'Metric': 'MatchNet Latent Embedding Vector', 'Wasserstein': w_dist, 'MeanDiff': w_dist})
        
    # Sort and Display
    comparisons.sort(key=lambda x: x['Wasserstein'], reverse=True)
    
    print(f"{'Metric':<35} | {'Wasserstein Dist':<18} | {'Absolute Mean Diff'}")
    print("-" * 75)
    for c in comparisons:
        print(f"{c['Metric']:<35} | {c['Wasserstein']:<18.4f} | {c['MeanDiff']:.4f}")
        
    print("\n[+] Phase B Analysis Complete.")

if __name__ == "__main__":
    compute_kul_fingerprint()
