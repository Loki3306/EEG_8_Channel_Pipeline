import os
import time
import numpy as np
import torch
from pathlib import Path
from scipy import signal
import multiprocessing as mp
import concurrent.futures
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score

# -------------------------------------------------------------------------
# CONSTANTS & CONFIGURATION
# -------------------------------------------------------------------------
SR = 128
EAR_CHANNEL_INDICES = [23, 31, 32, 40, 14, 22, 41, 49]
MAX_LAG_SEC = 0.400 
MAX_LAG_SAMPLES = int(MAX_LAG_SEC * SR)
SEQ_SAMPLES = int(3.0 * SR)
SEQ_HOP = int(0.5 * SR)

BROADBAND = (0.5, 8.0)
RIDGE_LAMBDA = 100.0
FORGETTING_FACTOR = 0.98

def apply_modulation_filter(env, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return signal.filtfilt(b, a, env, axis=1)

def create_toeplitz_features_pt(eeg, max_lag_samples):
    C, T = eeg.shape
    T_eff = T - max_lag_samples
    X = torch.zeros((T_eff, C * max_lag_samples), dtype=eeg.dtype, device=eeg.device)
    for tau in range(max_lag_samples):
        X[:, tau*C : (tau+1)*C] = eeg[:, tau : tau+T_eff].T
    return X

def get_pearsonr(x, y):
    x_m = x - np.mean(x)
    y_m = y - np.mean(y)
    num = np.dot(x_m, y_m)
    den = np.linalg.norm(x_m) * np.linalg.norm(y_m)
    return num / (den + 1e-8)

def prepare_subject_windows_continuous(cache_file, device):
    cached = torch.load(cache_file, map_location='cpu', weights_only=False)['raw']
    windows = []
    
    for tr in cached:
        eeg_raw = tr['eeg'].numpy()[EAR_CHANNEL_INDICES, :] 
        env_l_raw = tr['env_l'].numpy()
        env_r_raw = tr['env_r'].numpy()
        
        min_len = min(eeg_raw.shape[1], env_l_raw.shape[1])
        eeg_raw = eeg_raw[:, :min_len]
        env_l_raw = env_l_raw[:, :min_len]
        env_r_raw = env_r_raw[:, :min_len]
        
        eeg_f = apply_modulation_filter(eeg_raw, BROADBAND[0], BROADBAND[1], SR)
        env_l_f = apply_modulation_filter(env_l_raw, BROADBAND[0], BROADBAND[1], SR)
        env_r_f = apply_modulation_filter(env_r_raw, BROADBAND[0], BROADBAND[1], SR)
        
        # Standardized for the decoder
        eeg_f = (eeg_f - np.mean(eeg_f, axis=1, keepdims=True)) / (np.std(eeg_f, axis=1, keepdims=True) + 1e-8)
        env_l_f_z = (env_l_f - np.mean(env_l_f, axis=1, keepdims=True)) / (np.std(env_l_f, axis=1, keepdims=True) + 1e-8)
        env_r_f_z = (env_r_f - np.mean(env_r_f, axis=1, keepdims=True)) / (np.std(env_r_f, axis=1, keepdims=True) + 1e-8)
        
        eeg = torch.tensor(eeg_f, dtype=torch.float32, device=device)
        env_l = torch.tensor(env_l_f_z[0], dtype=torch.float32, device=device)
        env_r = torch.tensor(env_r_f_z[0], dtype=torch.float32, device=device)
        
        X_trial = create_toeplitz_features_pt(eeg, MAX_LAG_SAMPLES)
        T_eff = X_trial.shape[0]
        
        Y_l_eff = env_l[:T_eff]
        Y_r_eff = env_r[:T_eff]
        
        Y_l_raw_eff = env_l_raw[0, :T_eff]
        Y_r_raw_eff = env_r_raw[0, :T_eff]
        
        sp = tr['meta']['switch_points']
        current_spk = 'L'
        sp_idx = 0
        labels_eff = np.zeros(T_eff, dtype=int)
        for t in range(T_eff):
            if sp_idx < len(sp) and t >= sp[sp_idx][1]:
                current_spk = sp[sp_idx][0]
                sp_idx += 1
            labels_eff[t] = 1 if current_spk == 'L' else 0
            
        for seq_start in range(0, T_eff - SEQ_SAMPLES + 1, SEQ_HOP):
            seq_end = seq_start + SEQ_SAMPLES
            
            X_win = X_trial[seq_start:seq_end]
            Y_L_win = Y_l_eff[seq_start:seq_end]
            Y_R_win = Y_r_eff[seq_start:seq_end]
            Y_L_raw_win = Y_l_raw_eff[seq_start:seq_end]
            Y_R_raw_win = Y_r_raw_eff[seq_start:seq_end]
            
            win_labels = labels_eff[seq_start:seq_end]
            label = 1 if np.mean(win_labels) >= 0.5 else 0
            
            windows.append({
                'X': X_win,
                'Y_L': Y_L_win,
                'Y_R': Y_R_win,
                'Y_L_raw': Y_L_raw_win,
                'Y_R_raw': Y_R_raw_win,
                'label': label
            })
            
    return windows

class EmpiricalBayesianStateFilter:
    def __init__(self, mu_L, std_L, mu_R, std_R, rel_model):
        self.mu_L = mu_L
        self.var_L = std_L**2 + 1e-8
        self.mu_R = mu_R
        self.var_R = std_R**2 + 1e-8
        self.rel_model = rel_model
        
        self.p_L = 0.5  # Prior P(Attend Left)
        self.time_since_switch = 0
        self.current_inferred_state = -1 # Uninitialized
        
    def get_hazard(self):
        # Base hazard 0.001. Peaks around 20 windows (10 seconds)
        hazard = 0.001 + 0.05 / (1.0 + np.exp(-0.5 * (self.time_since_switch - 20)))
        return np.clip(hazard, 1e-6, 0.5)
        
    def update(self, e_t, env_var, env_sim):
        # 1. Prediction Step (Transition Hazard)
        A = self.get_hazard()
        p_L_pred = self.p_L * (1 - A) + (1 - self.p_L) * A
        p_R_pred = 1.0 - p_L_pred
        
        # 2. Observability Model (Heteroscedasticity)
        features = np.array([[env_var, env_sim]])
        # Predict probability of being observable
        try:
            r_t = self.rel_model.predict_proba(features)[0, 1]
        except:
            r_t = 0.5 # Fallback if model fails
            
        r_t = np.clip(r_t, 0.01, 1.0)
        
        # 3. Log-Likelihood Ratio
        # LLR = log P(e_t | L) - log P(e_t | R)
        ll_L = -0.5 * ((e_t - self.mu_L)**2 / self.var_L) - 0.5 * np.log(self.var_L)
        ll_R = -0.5 * ((e_t - self.mu_R)**2 / self.var_R) - 0.5 * np.log(self.var_R)
        
        LLR_base = ll_L - ll_R
        
        # Dilate variance = scale LLR by reliability
        LLR_t = r_t * LLR_base
        
        # 4. Bayesian Update Step in Log-Odds space
        prior_log_odds = np.log(p_L_pred / (p_R_pred + 1e-12))
        posterior_log_odds = prior_log_odds + LLR_t
        
        # Convert back to probability
        # clip to prevent overflow in exp
        posterior_log_odds = np.clip(posterior_log_odds, -50, 50)
        self.p_L = 1.0 / (1.0 + np.exp(-posterior_log_odds))
        
        # 5. Update Internal State Clock
        inferred_state = 1 if self.p_L > 0.5 else 0
        if inferred_state != self.current_inferred_state:
            self.time_since_switch = 0
            self.current_inferred_state = inferred_state
        else:
            self.time_since_switch += 1
            
        return self.p_L

def process_subject(cache_file, device_id):
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    subj_name = cache_file.stem.split('_')[0]
    
    windows = prepare_subject_windows_continuous(cache_file, device)
    if len(windows) < 400:
        return subj_name, None
        
    CALIB_WINDOWS = 240
    calib_set = windows[:CALIB_WINDOWS]
    track_set = windows[CALIB_WINDOWS:]
    
    F = windows[0]['X'].shape[1]
    
    # ---------------------------------------------------------
    # CALIBRATION PHASE
    # ---------------------------------------------------------
    Rxx_calib = torch.zeros((F, F), device=device)
    Rxy_calib = torch.zeros((F,), device=device)
    
    for w in calib_set:
        X = w['X']
        Y_true = w['Y_L'] if w['label'] == 1 else w['Y_R']
        Rxx_calib += X.T @ X
        Rxy_calib += X.T @ Y_true
        
    I = torch.eye(F, device=device)
    W_static = torch.linalg.solve(Rxx_calib + RIDGE_LAMBDA * I, Rxy_calib).cpu().numpy()
    
    # Fit Emissions and Reliability Model on Calibration Set
    calib_evidence = []
    calib_env_var = []
    calib_env_sim = []
    calib_observable = []
    calib_labels = []
    
    for w in calib_set:
        X_cpu = w['X'].cpu().numpy()
        YL_cpu = w['Y_L'].cpu().numpy()
        YR_cpu = w['Y_R'].cpu().numpy()
        YL_raw = w['Y_L_raw']
        YR_raw = w['Y_R_raw']
        label = w['label']
        
        preds = X_cpu @ W_static
        c_L = get_pearsonr(preds, YL_cpu)
        c_R = get_pearsonr(preds, YR_cpu)
        e_t = c_L - c_R
        
        env_var = np.var(YL_raw) + np.var(YR_raw) # RAW Envelope Variance
        env_sim = get_pearsonr(YL_cpu, YR_cpu) # Similarity
        
        c_true = c_L if label == 1 else c_R
        c_false = c_R if label == 1 else c_L
        observable = 1 if (c_true - c_false) > 0.0 else 0
        
        calib_evidence.append(e_t)
        calib_env_var.append(env_var)
        calib_env_sim.append(env_sim)
        calib_observable.append(observable)
        calib_labels.append(label)
        
    calib_evidence = np.array(calib_evidence)
    calib_labels = np.array(calib_labels)
    
    E_L = calib_evidence[calib_labels == 1]
    E_R = calib_evidence[calib_labels == 0]
    
    mu_L, std_L = np.mean(E_L), np.std(E_L)
    mu_R, std_R = np.mean(E_R), np.std(E_R)
    
    # Fallback if standard deviation is 0
    if std_L == 0: std_L = 0.1
    if std_R == 0: std_R = 0.1
    
    X_rel = np.column_stack((calib_env_var, calib_env_sim))
    y_rel = np.array(calib_observable)
    rel_model = LogisticRegression(class_weight='balanced')
    try:
        rel_model.fit(X_rel, y_rel)
    except:
        # Fallback if only 1 class in calibration
        class DummyModel:
            def predict_proba(self, X):
                return np.array([[0.5, 0.5]] * len(X))
        rel_model = DummyModel()
    
    bayesian_filter = EmpiricalBayesianStateFilter(mu_L, std_L, mu_R, std_R, rel_model)
    
    # Unsupervised Baseline Initialization
    Rxx_unsup = Rxx_calib.clone().cpu().numpy()
    Rxy_unsup = Rxy_calib.clone().cpu().numpy()
    W_unsup = W_static.copy()
    
    results_data = []
    
    for w in track_set:
        X_cpu = w['X'].cpu().numpy()
        YL_cpu = w['Y_L'].cpu().numpy()
        YR_cpu = w['Y_R'].cpu().numpy()
        YL_raw = w['Y_L_raw']
        YR_raw = w['Y_R_raw']
        label = w['label']
        
        # Static Decoder Evidence
        preds = X_cpu @ W_static
        c_L = get_pearsonr(preds, YL_cpu)
        c_R = get_pearsonr(preds, YR_cpu)
        e_t = c_L - c_R
        
        static_decision = 1 if e_t > 0 else 0
        
        # Empirical Bayesian Filter
        env_var = np.var(YL_raw) + np.var(YR_raw)
        env_sim = get_pearsonr(YL_cpu, YR_cpu)
        
        p_L = bayesian_filter.update(e_t, env_var, env_sim)
        bayes_decision = 1 if p_L >= 0.5 else 0
        
        # Unsupervised DD Baseline
        unsup_preds = X_cpu @ W_unsup
        uc_L = get_pearsonr(unsup_preds, YL_cpu)
        uc_R = get_pearsonr(unsup_preds, YR_cpu)
        unsup_decision = 1 if uc_L > uc_R else 0
        
        y_pseudo = YL_cpu if unsup_decision == 1 else YR_cpu
        Rxx_unsup = FORGETTING_FACTOR * Rxx_unsup + X_cpu.T @ X_cpu
        Rxy_unsup = FORGETTING_FACTOR * Rxy_unsup + X_cpu.T @ y_pseudo
        W_unsup = np.linalg.solve(Rxx_unsup + RIDGE_LAMBDA * np.eye(F), Rxy_unsup)
        
        results_data.append({
            'label': label,
            'static_decision': static_decision,
            'unsup_decision': unsup_decision,
            'bayes_decision': bayes_decision,
            'p_L': p_L
        })
        
    df = pd.DataFrame(results_data)
    
    # Metrics
    y_true = df['label'].values
    static_acc = accuracy_score(y_true, df['static_decision'])
    unsup_acc = accuracy_score(y_true, df['unsup_decision'])
    bayes_acc = accuracy_score(y_true, df['bayes_decision'])
    
    try:
        bayes_auroc = roc_auc_score(y_true, df['p_L'])
        bayes_brier = brier_score_loss(y_true, df['p_L'])
    except:
        bayes_auroc = 0.5
        bayes_brier = 0.25
        
    metrics = {
        'subject': subj_name,
        'static_acc': static_acc,
        'unsup_acc': unsup_acc,
        'bayes_acc': bayes_acc,
        'bayes_auroc': bayes_auroc,
        'bayes_brier': bayes_brier,
        'total': len(track_set)
    }
    
    return metrics

def main():
    mp.set_start_method('spawn', force=True)
    print("=======================================================")
    print(" PHASE 154: EMPIRICAL BAYESIAN STATE FILTER")
    print("=======================================================\n")
    
    cache_dir = Path('/kaggle/working/multiband_cache')
    possible_paths = [
        Path('/kaggle/input/datasets/lowkieee/multiband-cache/kaggle/working/multiband_cache'),
        Path('/kaggle/input/multiband-cache/kaggle/working/multiband_cache'),
        Path('/kaggle/input/aasd-universal-cache-v1/kaggle/working/multiband_cache'),
        cache_dir
    ]
    for p in possible_paths:
        if p.exists() and len(list(p.glob('*_multiband.pt'))) > 0:
            cache_dir = p
            break
            
    cache_files = sorted(list(cache_dir.glob('*_multiband.pt')))
    num_gpus = torch.cuda.device_count()
    num_workers = min(mp.cpu_count(), num_gpus if num_gpus > 0 else mp.cpu_count())
    
    all_metrics = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_subject, cf, idx % max(1, num_gpus)): cf for idx, cf in enumerate(cache_files)}
        for future in concurrent.futures.as_completed(futures):
            m = future.result()
            if m is not None:
                all_metrics.append(m)
                print(f"[{m['subject']:3s}] Static: {m['static_acc']*100:4.1f}% | Unsup DD: {m['unsup_acc']*100:4.1f}% | Empirical Bayes: {m['bayes_acc']*100:4.1f}% | Bayes AUROC: {m['bayes_auroc']:.3f} | Brier: {m['bayes_brier']:.3f}")

    if all_metrics:
        mean_static = np.mean([m['static_acc'] for m in all_metrics])
        mean_unsup = np.mean([m['unsup_acc'] for m in all_metrics])
        mean_bayes = np.mean([m['bayes_acc'] for m in all_metrics])
        mean_auroc = np.mean([m['bayes_auroc'] for m in all_metrics])
        mean_brier = np.mean([m['bayes_brier'] for m in all_metrics])
        
        print("\n=======================================================")
        print(" FINAL RESULTS (MEAN ACROSS SUBJECTS)")
        print("=======================================================")
        print(f" Static Decoder Baseline : {mean_static*100:.2f}%")
        print(f" Unsupervised DD Tracker : {mean_unsup*100:.2f}%")
        print(f" Empirical Bayesian HMM  : {mean_bayes*100:.2f}%")
        print(f" Empirical Bayes AUROC   : {mean_auroc:.3f}")
        print(f" Empirical Bayes Brier   : {mean_brier:.3f}")
        print("=======================================================")

if __name__ == '__main__':
    main()
