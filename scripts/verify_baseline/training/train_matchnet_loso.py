import argparse
import sys
import os
import json
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import psutil
import gc
from pathlib import Path
from copy import deepcopy
from scipy.signal import butter, filtfilt
from torch.utils.data import TensorDataset, DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.matchnet import ContrastiveMatchNet, contrastive_loss, anchored_contrastive_loss, dcca_loss
from baselines.ridge_aad import load_subject_examples, subject_files, iter_leave_one_subject_out

FS = 64
DECISION_WINDOW_SEC = 10
TRAIN_WINDOW_SEC = 5
TRAIN_HOP_SEC = 2
NUM_BANDS = 28

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2, axis=0):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data, axis=axis)
    return y

def normalize_array(arr):
    arr = arr - arr.mean(axis=0, keepdims=True)
    scale = arr.std(axis=0, keepdims=True) + 1e-12
    return arr / scale

def normalize_array_global(arr):
    arr = arr - arr.mean(axis=0, keepdims=True)
    scale = arr.std() + 1e-12
    return arr / scale

def get_mapping_data(audio_rep="gammatone", audio_env_file=""):
    base_dir = Path("/kaggle/input")
    
    # 1. Find audio_mapping.json
    map_files = list(base_dir.rglob("audio_mapping.json"))
    if map_files:
        map_file = map_files[0]
    else:
        map_file = REPO_ROOT / "data" / "audio_mapping.json"
        
    # 2. Find envelopes pkl
    if audio_env_file:
        env_file = Path(audio_env_file)
    else:
        search_pattern = "*wavlm*.pkl" if audio_rep == "wavlm" else "*gammatone*.pkl"
        pkl_files = list(base_dir.rglob(search_pattern))
        if not pkl_files:
            pkl_files = list(base_dir.rglob("*.pkl"))
            
        if pkl_files:
            env_file = pkl_files[0]
        else:
            default_name = "wavlm_features.pkl" if audio_rep == "wavlm" else "gammatone_envelopes.pkl"
            env_file = REPO_ROOT / "data" / default_name
        
    print(f"Using map file: {map_file}")
    print(f"Using env file: {env_file}")
    
    with open(map_file, 'r') as f:
        mapping = json.load(f)
    with open(env_file, 'rb') as f:
        envelopes = pickle.load(f)
    return mapping, envelopes

def prepare_dataset(examples, channels, lowcut, highcut, subject_id, mapping, envelopes, exclude_audio_files=None, audio_layer_idx=0, lag_sec=0.0):
    X = []
    Y_A = []
    Y_B = []
    
    sub_key = subject_id.replace("_data_preproc", "")
    shift_samples = int(round(lag_sec * FS))
    
    for i, ex in enumerate(examples):
        trial_key = f"trial_{i}"
        
        if sub_key in mapping and trial_key in mapping[sub_key]:
            fname_a = mapping[sub_key][trial_key]["wavA"]["filename"]
            fname_b = mapping[sub_key][trial_key]["wavB"]["filename"]
            
            if exclude_audio_files is not None and (fname_a in exclude_audio_files or fname_b in exclude_audio_files):
                continue
            
            env_a = envelopes[fname_a]
            env_b = envelopes[fname_b]
            
            # DTU convention: wavA is ALWAYS attended, wavB is ALWAYS unattended
            env_attended = env_a
            env_unattended = env_b
            
            if len(env_attended.shape) == 3:
                env_attended = env_attended[audio_layer_idx]
                env_unattended = env_unattended[audio_layer_idx]
        else:
            print(f"Warning: Missing mapping for {sub_key} {trial_key}")
            continue
            
        eeg = ex.eeg[:, channels].T
        eeg = butter_bandpass_filter(eeg, lowcut, highcut, FS, axis=1)
        x_norm = normalize_array(eeg.T).T 
            
        env_attended = normalize_array(env_attended.T).T
        env_unattended = normalize_array(env_unattended.T).T
        
        # Temporal alignment with neural lag:
        # If lag_sec > 0 (+250ms), EEG lags acoustic stimulus.
        # EEG at sample t reflects speech at sample t - shift_samples.
        # Aligns EEG[shift_samples:] with Audio[:-shift_samples].
        if shift_samples > 0:
            if x_norm.shape[1] > shift_samples and env_attended.shape[1] > shift_samples:
                x_norm = x_norm[:, shift_samples:]
                env_attended = env_attended[:, :-shift_samples]
                env_unattended = env_unattended[:, :-shift_samples]
        elif shift_samples < 0:
            abs_shift = abs(shift_samples)
            if x_norm.shape[1] > abs_shift and env_attended.shape[1] > abs_shift:
                x_norm = x_norm[:, :-abs_shift]
                env_attended = env_attended[:, abs_shift:]
                env_unattended = env_unattended[:, abs_shift:]
        
        min_len = min(x_norm.shape[1], env_attended.shape[1])
        x_norm = x_norm[:, :min_len]
        env_attended = env_attended[:, :min_len]
        env_unattended = env_unattended[:, :min_len]
        
        X.append(x_norm.astype(np.float32))
        Y_A.append(env_attended.astype(np.float32))
        Y_B.append(env_unattended.astype(np.float32))
        
    return X, Y_A, Y_B

def select_top_channels_from_train(subject_examples, train_paths, mapping, envelopes, num_channels, lowcut=1.0, highcut=6.0, audio_layer_idx=0):
    """
    Ranks all 64 DTU channels using ONLY the training subjects (zero leakage into held-out subject).
    Computes absolute correlation between each EEG channel and attended speech envelope.
    """
    if num_channels >= 64:
        return list(range(64)), np.ones(64)
        
    channel_scores = np.zeros(64, dtype=np.float64)
    channel_counts = np.zeros(64, dtype=np.int32)
    
    for p in train_paths:
        sub_key = p.stem.replace("_data_preproc", "")
        exs = subject_examples[str(p)]
        # Sample up to 10 trials per subject to compute fast correlation ranking
        sample_exs = exs[:10]
        for i, ex in enumerate(sample_exs):
            trial_key = f"trial_{i}"
            if sub_key in mapping and trial_key in mapping[sub_key]:
                fname_a = mapping[sub_key][trial_key]["wavA"]["filename"]
                if fname_a not in envelopes:
                    continue
                env_a = envelopes[fname_a]
                if len(env_a.shape) == 3:
                    env_a = env_a[audio_layer_idx]
                
                # Speech envelope energy across bands: mean across bands -> [T]
                env_1d = env_a.mean(axis=0)
                env_1d = butter_bandpass_filter(env_1d, lowcut, highcut, FS, axis=0)
                env_1d = (env_1d - env_1d.mean()) / (env_1d.std() + 1e-12)
                
                # EEG for all 64 channels: [64, T]
                eeg_all = ex.eeg.T # [64, T]
                eeg_all = butter_bandpass_filter(eeg_all, lowcut, highcut, FS, axis=1)
                
                min_len = min(eeg_all.shape[1], len(env_1d))
                eeg_all = eeg_all[:, :min_len]
                env_sub = env_1d[:min_len]
                
                # Normalize EEG per channel
                eeg_mean = eeg_all.mean(axis=1, keepdims=True)
                eeg_std = eeg_all.std(axis=1, keepdims=True) + 1e-12
                eeg_norm = (eeg_all - eeg_mean) / eeg_std
                
                # Pearson correlation with speech envelope for each channel
                corrs = np.abs((eeg_norm * env_sub[None, :]).mean(axis=1)) # [64]
                channel_scores += corrs
                channel_counts += 1
                
    avg_scores = channel_scores / np.maximum(channel_counts, 1)
    ranked_indices = np.argsort(-avg_scores).tolist()
    selected = ranked_indices[:num_channels]
    return selected, avg_scores

def chunk_trial(x, ya, yb, window_sec, hop_sec):
    """Chunks a single trial into smaller overlapping windows for training."""
    win_samples = int(window_sec * FS)
    hop_samples = int(hop_sec * FS)
    
    chunks_x, chunks_ya, chunks_yb = [], [], []
    start = 0
    while start + win_samples <= x.shape[1]:
        end = start + win_samples
        chunks_x.append(x[:, start:end])
        chunks_ya.append(ya[:, start:end])
        chunks_yb.append(yb[:, start:end])
        start += hop_samples
        
    return chunks_x, chunks_ya, chunks_yb

def pearson_corr(x, y, dim=1):
    x_centered = x - x.mean(dim=dim, keepdim=True)
    y_centered = y - y.mean(dim=dim, keepdim=True)
    cov = (x_centered * y_centered).sum(dim=dim)
    var_x = (x_centered ** 2).sum(dim=dim)
    var_y = (y_centered ** 2).sum(dim=dim)
    return cov / torch.sqrt(var_x * var_y + 1e-8)

def evaluate_model(model, X, Y_A, Y_B, device, window_sec=10, zero_eeg=False, shuffle_labels=False, metric="cosine", shuffle_eeg_time=False, shuffle_audio_time=False, permute_spatial=False, swap_ab=False, use_absolute_scoring=False):
    """
    Evaluates the model using non-overlapping windows.
    Decision rule: metric(Z_eeg, Z_A) > metric(Z_eeg, Z_B)
    """
    model.eval()
    window_samples = int(window_sec * FS)
    n_correct = 0.0
    n_total = 0
    
    np.random.seed(42)
    shuffle_indices = np.random.permutation(len(X))
    while len(X) > 1 and np.any(shuffle_indices == np.arange(len(X))):
        shuffle_indices = np.random.permutation(len(X))
    
    with torch.no_grad():
        for i in range(len(X)):
            x_np = X[i]
            
            if shuffle_labels:
                shuf_idx = shuffle_indices[i]
                ya_np = Y_A[shuf_idx]
                yb_np = Y_B[shuf_idx]
            else:
                ya_np = Y_A[i]
                yb_np = Y_B[i]
            
            start = 0
            while start + window_samples <= x_np.shape[1]:
                end = start + window_samples
                
                x_chunk = torch.from_numpy(x_np[:, start:end]).unsqueeze(0).to(device, dtype=torch.float32)
                if zero_eeg:
                    x_chunk = torch.zeros_like(x_chunk)
                    
                ya_chunk = torch.from_numpy(ya_np[:, start:end]).unsqueeze(0).to(device, dtype=torch.float32)
                yb_chunk = torch.from_numpy(yb_np[:, start:end]).unsqueeze(0).to(device, dtype=torch.float32)
                
                if swap_ab:
                    ya_chunk, yb_chunk = yb_chunk, ya_chunk
                    
                if shuffle_eeg_time:
                    perm = torch.randperm(x_chunk.shape[-1], device=device)
                    x_chunk = x_chunk[..., perm]
                if shuffle_audio_time:
                    perm = torch.randperm(ya_chunk.shape[-1], device=device)
                    ya_chunk = ya_chunk[..., perm]
                    yb_chunk = yb_chunk[..., perm]
                if permute_spatial:
                    perm = torch.randperm(x_chunk.shape[1], device=device)
                    x_chunk = x_chunk[:, perm, :]
                
                z_eeg, z_a, z_b = model(x_chunk, ya_chunk, yb_chunk)
                
                if metric == "pearson":
                    sim_a = pearson_corr(z_eeg, z_a, dim=1).mean().item()
                    sim_b = pearson_corr(z_eeg, z_b, dim=1).mean().item()
                else:
                    sim_a = F.cosine_similarity(z_eeg, z_a, dim=1).mean().item()
                    sim_b = F.cosine_similarity(z_eeg, z_b, dim=1).mean().item()
                
                if use_absolute_scoring:
                    sim_a = abs(sim_a)
                    sim_b = abs(sim_b)
                
                if swap_ab:
                    if sim_b > sim_a: n_correct += 1.0
                    elif sim_a == sim_b: n_correct += 0.5
                else:
                    if sim_a > sim_b: n_correct += 1.0
                    elif sim_a == sim_b: n_correct += 0.5
                    
                n_total += 1
                start += window_samples
                
    return n_correct, n_total

def evaluate_evidence_aggregation(model, X, Y_A, Y_B, device, metric="pearson", use_absolute_scoring=False):
    """
    Evaluates model across multiple decision window lengths [1s, 2s, 5s, 10s, 15s, 20s, 25s, 30s, 35s, 40s].
    Computes:
      1. 1s-base sub-window similarity accumulation (D = sum_t (s_A(t) - s_B(t)))
      2. 1s-base majority voting
      3. 5s-base sub-window similarity accumulation
      4. Direct independent window classification (non-overlapping W-second windows fed directly into model)
    """
    model.eval()
    
    trial_sub_1s = []  # list of (d_list, vote_list)
    trial_sub_5s = []  # list of (d_list, vote_list)
    
    sample_rate = FS
    samples_1s = int(1 * sample_rate)
    samples_5s = int(5 * sample_rate)
    
    with torch.no_grad():
        for i in range(len(X)):
            x_np = X[i]
            ya_np = Y_A[i]
            yb_np = Y_B[i]
            trial_len = x_np.shape[1]
            
            # --- 1-second atomic sub-windows ---
            d_1s = []
            v_1s = []
            start = 0
            while start + samples_1s <= trial_len:
                end = start + samples_1s
                x_chunk = torch.from_numpy(x_np[:, start:end]).unsqueeze(0).to(device, dtype=torch.float32)
                ya_chunk = torch.from_numpy(ya_np[:, start:end]).unsqueeze(0).to(device, dtype=torch.float32)
                yb_chunk = torch.from_numpy(yb_np[:, start:end]).unsqueeze(0).to(device, dtype=torch.float32)
                
                z_eeg, z_a, z_b = model(x_chunk, ya_chunk, yb_chunk)
                if metric == "pearson":
                    sa = pearson_corr(z_eeg, z_a, dim=1).mean().item()
                    sb = pearson_corr(z_eeg, z_b, dim=1).mean().item()
                else:
                    sa = F.cosine_similarity(z_eeg, z_a, dim=1).mean().item()
                    sb = F.cosine_similarity(z_eeg, z_b, dim=1).mean().item()
                
                if use_absolute_scoring:
                    sa, sb = abs(sa), abs(sb)
                    
                diff = sa - sb
                d_1s.append(diff)
                v_1s.append(1.0 if diff > 0 else (0.5 if diff == 0 else 0.0))
                start += samples_1s
                
            trial_sub_1s.append((d_1s, v_1s))
            
            # --- 5-second atomic sub-windows ---
            d_5s = []
            v_5s = []
            start = 0
            while start + samples_5s <= trial_len:
                end = start + samples_5s
                x_chunk = torch.from_numpy(x_np[:, start:end]).unsqueeze(0).to(device, dtype=torch.float32)
                ya_chunk = torch.from_numpy(ya_np[:, start:end]).unsqueeze(0).to(device, dtype=torch.float32)
                yb_chunk = torch.from_numpy(yb_np[:, start:end]).unsqueeze(0).to(device, dtype=torch.float32)
                
                z_eeg, z_a, z_b = model(x_chunk, ya_chunk, yb_chunk)
                if metric == "pearson":
                    sa = pearson_corr(z_eeg, z_a, dim=1).mean().item()
                    sb = pearson_corr(z_eeg, z_b, dim=1).mean().item()
                else:
                    sa = F.cosine_similarity(z_eeg, z_a, dim=1).mean().item()
                    sb = F.cosine_similarity(z_eeg, z_b, dim=1).mean().item()
                
                if use_absolute_scoring:
                    sa, sb = abs(sa), abs(sb)
                    
                diff = sa - sb
                d_5s.append(diff)
                v_5s.append(1.0 if diff > 0 else (0.5 if diff == 0 else 0.0))
                start += samples_5s
                
            trial_sub_5s.append((d_5s, v_5s))

    # Evaluate aggregations across target decision windows
    windows_all = [1, 2, 5, 10, 15, 20, 25, 30, 35, 40]
    results = {}
    
    for w in windows_all:
        # 1. 1s-accumulated similarity & 1s-majority vote
        c_accum_1s, n_accum_1s = 0.0, 0
        c_vote_1s = 0.0
        m_1s = w # w chunks of 1s
        for d_list, v_list in trial_sub_1s:
            b_start = 0
            while b_start + m_1s <= len(d_list):
                block_d = d_list[b_start : b_start + m_1s]
                block_v = v_list[b_start : b_start + m_1s]
                sum_d = sum(block_d)
                if sum_d > 0: c_accum_1s += 1.0
                elif sum_d == 0: c_accum_1s += 0.5
                
                sum_v = sum(block_v)
                if sum_v > m_1s / 2.0: c_vote_1s += 1.0
                elif sum_v == m_1s / 2.0: c_vote_1s += 0.5
                
                n_accum_1s += 1
                b_start += m_1s
                
        acc_accum_1s = c_accum_1s / max(n_accum_1s, 1)
        acc_vote_1s = c_vote_1s / max(n_accum_1s, 1)
        
        # 2. 5s-accumulated similarity (for w >= 5 and w % 5 == 0)
        acc_accum_5s = None
        if w >= 5 and w % 5 == 0:
            m_5s = w // 5
            c_accum_5s, n_accum_5s = 0.0, 0
            for d_list, _ in trial_sub_5s:
                b_start = 0
                while b_start + m_5s <= len(d_list):
                    block_d = d_list[b_start : b_start + m_5s]
                    sum_d = sum(block_d)
                    if sum_d > 0: c_accum_5s += 1.0
                    elif sum_d == 0: c_accum_5s += 0.5
                    n_accum_5s += 1
                    b_start += m_5s
            acc_accum_5s = c_accum_5s / max(n_accum_5s, 1)
            
        # 3. Direct independent window evaluation
        c_direct, n_direct = 0.0, 0
        w_samples = int(w * sample_rate)
        with torch.no_grad():
            for i in range(len(X)):
                x_np = X[i]
                ya_np = Y_A[i]
                yb_np = Y_B[i]
                start = 0
                while start + w_samples <= x_np.shape[1]:
                    end = start + w_samples
                    x_chunk = torch.from_numpy(x_np[:, start:end]).unsqueeze(0).to(device, dtype=torch.float32)
                    ya_chunk = torch.from_numpy(ya_np[:, start:end]).unsqueeze(0).to(device, dtype=torch.float32)
                    yb_chunk = torch.from_numpy(yb_np[:, start:end]).unsqueeze(0).to(device, dtype=torch.float32)
                    
                    z_eeg, z_a, z_b = model(x_chunk, ya_chunk, yb_chunk)
                    if metric == "pearson":
                        sa = pearson_corr(z_eeg, z_a, dim=1).mean().item()
                        sb = pearson_corr(z_eeg, z_b, dim=1).mean().item()
                    else:
                        sa = F.cosine_similarity(z_eeg, z_a, dim=1).mean().item()
                        sb = F.cosine_similarity(z_eeg, z_b, dim=1).mean().item()
                        
                    if use_absolute_scoring:
                        sa, sb = abs(sa), abs(sb)
                        
                    if sa > sb: c_direct += 1.0
                    elif sa == sb: c_direct += 0.5
                    
                    n_direct += 1
                    start += w_samples
                    
        acc_direct = c_direct / max(n_direct, 1)
        
        results[w] = {
            "accum_1s": acc_accum_1s,
            "vote_1s": acc_vote_1s,
            "accum_5s": acc_accum_5s,
            "direct": acc_direct,
            "decisions": n_accum_1s
        }
    return results

def run_lag_sweep(model, examples, channels, lowcut, highcut, subject_id, mapping, envelopes, device, audio_layer_idx=0, use_absolute_scoring=False):
    """
    Sweeps fixed temporal lags between EEG and Speech:
    [-1000ms, -750ms, -500ms, -250ms, 0ms, +150ms, +250ms, +500ms, +750ms, +1000ms]
    Evaluates at 10s decision window to isolate the neural latency profile.
    """
    lags_sec = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.15, 0.25, 0.5, 0.75, 1.0]
    sweep_results = {}
    print("\n" + "="*65)
    print(f"[FIXED NEURAL/AUDIO LAG SWEEP (10s Window) - Subject: {subject_id}]")
    print("="*65)
    print(f" {'Lag (ms)':>10} | {'Shift (samples)':>15} | {'Accuracy':>10} | {'Decisions':>10}")
    print("-" * 65)
    
    for lag in lags_sec:
        X_lag, YA_lag, YB_lag = prepare_dataset(
            examples, channels, lowcut, highcut, subject_id, mapping, envelopes, 
            audio_layer_idx=audio_layer_idx, lag_sec=lag
        )
        nc, nt = evaluate_model(model, X_lag, YA_lag, YB_lag, device, window_sec=10, metric="pearson", use_absolute_scoring=use_absolute_scoring)
        acc = nc / max(nt, 1)
        shift_samples = int(round(lag * FS))
        lag_ms = int(lag * 1000)
        sweep_results[lag_ms] = acc
        print(f" {lag_ms:+9d} ms | {shift_samples:+14d} | {acc*100:9.2f}% | {nt:10d}")
        
    print("="*65)
    best_lag = max(sweep_results.keys(), key=lambda k: sweep_results[k])
    base_acc = sweep_results[0]
    print(f"  -> Baseline (0 ms): {base_acc*100:.2f}%")
    print(f"  -> Optimal Lag:     {best_lag:+d} ms ({sweep_results[best_lag]*100:.2f}%, delta: {(sweep_results[best_lag] - base_acc)*100:+.2f}%)")
    print("="*65)
    return sweep_results

class ChunkDataset(torch.utils.data.Dataset):
    def __init__(self, X_full, YA_full, YB_full, chunk_indices, Subj_full=None):
        self.X_full = X_full
        self.YA_full = YA_full
        self.YB_full = YB_full
        self.chunk_indices = chunk_indices
        self.Subj_full = Subj_full
        
    def __len__(self):
        return len(self.chunk_indices)
        
    def __getitem__(self, idx):
        trial_idx, start, end = self.chunk_indices[idx]
        x = self.X_full[trial_idx][:, start:end]
        ya = self.YA_full[trial_idx][:, start:end]
        yb = self.YB_full[trial_idx][:, start:end]
        
        if self.Subj_full is not None:
            subj = torch.tensor(self.Subj_full[trial_idx], dtype=torch.long)
            return x, ya, yb, subj
        return x, ya, yb

def train_matchnet_loso(eeg_model="eegnet", channels=None, num_channels=8, rank_channels=False, lowcut=1.0, highcut=6.0, batch_size=128, num_workers=2, subjects_to_run=None, loss_type="contrastive", lambda_align=0.5, align_target=0.1, augment_sign_flip=False, use_dann=False, use_temporal_transport=False, audio_rep="gammatone", audio_env_file="", audio_layer_idx=0, hard_negative_prob=0.0, lag_sec=0.0, sweep_lags=False, eval_only=False, checkpoint_dir="checkpoints"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if audio_rep == "wavlm":
        print("Forcing num_workers=0 for WavLM to prevent multiprocessing RAM explosion.")
        num_workers = 0
        
    default_wearable = [0, 33, 6, 41, 22, 59, 15, 52]
    
    all_paths = subject_files()
    if not all_paths:
        print("No subjects found.")
        return
        
    subject_examples = {str(p): load_subject_examples(p) for p in all_paths}
    folds = list(iter_leave_one_subject_out(all_paths))
    
    if subjects_to_run:
        folds = [f for f in folds if f[0].stem in subjects_to_run]
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    grand_summary = {}
    
    print(f"\n=================================================================")
    print(f" MATCHNET EXPERIMENT RUNNER")
    print(f" Model: {eeg_model.upper()} | Audio: {audio_rep.upper()} | Device: {device}")
    print(f" Loss: {loss_type} | Lag: {lag_sec*1000:+.0f} ms | Eval Only: {eval_only}")
    print(f" Folds to evaluate: {len(folds)} subject(s)")
    print(f"=================================================================\n")
    
    for held_out_path, train_paths in folds:
        held_out_key = str(held_out_path)
        sub_name = held_out_path.stem
        print(f"\n{'='*60}")
        print(f" FOLD: Held-out Subject {sub_name}")
        print(f" Pre-fold RAM: {psutil.virtual_memory().percent}% ({psutil.virtual_memory().used / 1e9:.2f} GB used)")
        print(f"{'='*60}")
        
        # Load heavy audio features inside the loop so they can be deleted after extraction
        mapping, envelopes = get_mapping_data(audio_rep, audio_env_file)
        
        # Determine channels for this fold
        if channels is not None and len(channels) > 0:
            fold_channels = list(channels)
            print(f"  [Channel Setup]: Using explicit user channels ({len(fold_channels)} ch): {fold_channels}")
        elif rank_channels or (num_channels in [16, 32, 64] and channels is None):
            print(f"  [Channel Setup]: Ranking channels on TRAINING subjects only ({len(train_paths)} subjects, zero test leakage)...")
            fold_channels, scores = select_top_channels_from_train(
                subject_examples, train_paths, mapping, envelopes, num_channels, lowcut, highcut, audio_layer_idx
            )
            print(f"  [Channel Setup]: Selected Top {num_channels} channels: {fold_channels}")
        else:
            fold_channels = default_wearable
            print(f"  [Channel Setup]: Using default 8-channel wearable montage: {fold_channels}")
            
        test_exs = subject_examples[held_out_key]
        
        X_va_full = []
        YA_va_full = []
        YB_va_full = []
        
        X_tr_full, YA_tr_full, YB_tr_full = [], [], []
        Subj_tr_full = []
        Subj_va_full = []
        curr_id = 0
        subject_id_map = {}
        
        for p in train_paths:
            if p.stem not in subject_id_map:
                subject_id_map[p.stem] = curr_id
                curr_id += 1
            subj_id = subject_id_map[p.stem]
            
            # 1. Trial-Level Split: Shuffle the trials for this subject
            exs = list(subject_examples[str(p)])
            np.random.seed(42)  # Strict global seed to eliminate audio overlap
            np.random.shuffle(exs)
            
            val_split_num = int(0.1 * len(exs))
            if val_split_num == 0 and len(exs) > 0:
                val_split_num = 1
                
            val_exs = exs[:val_split_num]
            train_exs = exs[val_split_num:]
            
            # Extract Training Trials
            tX, tYA, tYB = prepare_dataset(
                train_exs, fold_channels, lowcut, highcut, p.stem, mapping, envelopes, 
                audio_layer_idx=audio_layer_idx, lag_sec=lag_sec
            )
            X_tr_full.extend(tX)
            YA_tr_full.extend(tYA)
            YB_tr_full.extend(tYB)
            Subj_tr_full.extend([subj_id] * len(tX))
            
            # Extract Validation Trials
            vX, vYA, vYB = prepare_dataset(
                val_exs, fold_channels, lowcut, highcut, p.stem, mapping, envelopes, 
                audio_layer_idx=audio_layer_idx, lag_sec=lag_sec
            )
            X_va_full.extend(vX)
            YA_va_full.extend(vYA)
            YB_va_full.extend(vYB)
            Subj_va_full.extend([subj_id] * len(vX))
            
        # Extract Test Trials
        X_te_full, YA_te_full, YB_te_full = prepare_dataset(
            test_exs, fold_channels, lowcut, highcut, sub_name, mapping, envelopes, 
            audio_layer_idx=audio_layer_idx, lag_sec=lag_sec
        )
        
        # Free envelopes dictionary
        del envelopes
        gc.collect()
        
        # Convert test to torch
        X_te_t = [torch.from_numpy(x) for x in X_te_full]
        YA_te_t = [torch.from_numpy(x) for x in YA_te_full]
        YB_te_t = [torch.from_numpy(x) for x in YB_te_full]
        
        # Model initialization
        audio_channels = 768 if audio_rep == "wavlm" else 28
        model = ContrastiveMatchNet(
            eeg_model_type=eeg_model, 
            eeg_channels=len(fold_channels), 
            audio_channels=audio_channels,
            latent_dim=64,
            use_temporal_transport=use_temporal_transport,
            audio_model_type=audio_rep
        ).to(device)
        
        best_path = Path(checkpoint_dir) / f"matchnet_fold_{sub_name}_best.pth"
        
        if eval_only:
            if not best_path.exists():
                print(f"Warning: Checkpoint {best_path} not found. Searching for any fold checkpoint...")
                ckpts = list(Path(checkpoint_dir).glob(f"*{sub_name}*.pth"))
                if ckpts:
                    best_path = ckpts[0]
                else:
                    raise FileNotFoundError(f"No checkpoint found for {sub_name} in {checkpoint_dir}")
            print(f"Loading checkpoint for evaluation: {best_path}")
            model.load_state_dict(torch.load(best_path, map_location=device))
        else:
            # Training Phase
            X_tr_full = [torch.from_numpy(x) for x in X_tr_full]
            YA_tr_full = [torch.from_numpy(x) for x in YA_tr_full]
            YB_tr_full = [torch.from_numpy(x) for x in YB_tr_full]
            
            X_va_full = [torch.from_numpy(x) for x in X_va_full]
            YA_va_full = [torch.from_numpy(x) for x in YA_va_full]
            YB_va_full = [torch.from_numpy(x) for x in YB_va_full]
            
            chunk_indices = []
            win_samples = int(TRAIN_WINDOW_SEC * FS)
            hop_samples = int(TRAIN_HOP_SEC * FS)
            for i in range(len(X_tr_full)):
                trial_len = X_tr_full[i].shape[1]
                start = 0
                while start + win_samples <= trial_len:
                    chunk_indices.append((i, start, start + win_samples))
                    start += hop_samples
                
            train_dataset = ChunkDataset(X_tr_full, YA_tr_full, YB_tr_full, chunk_indices, Subj_tr_full if use_dann else None)
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=num_workers,
                pin_memory=(num_workers > 0)
            )
            
            optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
            
            best_val_acc = 0.0
            best_weights = deepcopy(model.state_dict())
            patience = 5
            epochs_no_improve = 0
            
            print(f"Training on {len(chunk_indices)} chunks ({TRAIN_WINDOW_SEC}s) | Batch Size: {batch_size}...")
            
            for epoch in range(30):
                model.train()
                train_loss, train_sa, train_sb = 0.0, 0.0, 0.0
                
                for batch in train_loader:
                    if use_dann:
                        bx, bya, byb, b_subj = batch
                        b_subj = b_subj.to(device, non_blocking=True)
                    else:
                        bx, bya, byb = batch
                        
                    bx = bx.to(device, non_blocking=True)
                    bya = bya.to(device, non_blocking=True)
                    byb = byb.to(device, non_blocking=True)
                    
                    if augment_sign_flip:
                        sign = torch.randint(0, 2, (bx.size(0), 1, 1), device=device).float() * 2.0 - 1.0
                        bx = bx * sign
                        
                    if hard_negative_prob > 0.0 and torch.rand(1).item() < hard_negative_prob:
                        shift_amount = torch.randint(64, bya.size(-1) - 64, (1,)).item()
                        byb = torch.roll(bya, shifts=shift_amount, dims=-1)
                    
                    optimizer.zero_grad()
                    with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                        z_eeg, z_a, z_b = model(bx, bya, byb)
                        if loss_type == "anchored":
                            loss, sa, sb = anchored_contrastive_loss(z_eeg, z_a, z_b, margin=0.1, lambda_align=lambda_align, align_target=align_target)
                        elif loss_type == "absolute":
                            sim_a = F.cosine_similarity(z_eeg, z_a, dim=1).mean(dim=1)
                            sim_b = F.cosine_similarity(z_eeg, z_b, dim=1).mean(dim=1)
                            loss = F.relu(0.1 - (torch.abs(sim_a) - torch.abs(sim_b))).mean()
                            sa = torch.abs(sim_a).mean()
                            sb = torch.abs(sim_b).mean()
                        elif loss_type == "dcca":
                            B, D, T = z_eeg.shape
                            z_eeg_flat = z_eeg.transpose(1, 2).reshape(B * T, D)
                            z_a_flat = z_a.transpose(1, 2).reshape(B * T, D)
                            loss, sa, sb = dcca_loss(z_eeg_flat, z_a_flat)
                        else:
                            loss, sa, sb = contrastive_loss(z_eeg, z_a, z_b, margin=0.1)
                    
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                        
                    train_loss += loss.item()
                    train_sa += sa.item()
                    train_sb += sb.item()
                    
                nc_va, nt_va = evaluate_model(model, X_va_full, YA_va_full, YB_va_full, device, window_sec=10, use_absolute_scoring=(loss_type == "absolute"))
                val_acc = nc_va / max(nt_va, 1)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_weights = deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    
                num_batches = max(len(train_loader), 1)
                print(f"  Epoch {epoch+1:02d}/30 | Loss: {train_loss/num_batches:.4f} (sA: {train_sa/num_batches:.3f}, sB: {train_sb/num_batches:.3f}) | Val Acc (10s): {val_acc*100:.2f}% | Patience: {epochs_no_improve}/{patience}")
                if epochs_no_improve >= patience:
                    break
                    
            torch.save(best_weights, best_path)
            model.load_state_dict(best_weights)
            
            # Clean up training data to free RAM
            del X_tr_full, YA_tr_full, YB_tr_full, X_va_full, YA_va_full, YB_va_full, train_dataset, train_loader
            gc.collect()

        # Multi-Window Decision Aggregation Evaluation
        print(f"\n  [EVALUATION: Multi-Window Evidence Aggregation for {sub_name}]")
        multi_win_results = evaluate_evidence_aggregation(
            model, X_te_full, YA_te_full, YB_te_full, device, metric="pearson", use_absolute_scoring=(loss_type == "absolute")
        )
        
        # Display Decision Window Breakdown Table
        print("\n" + "="*88)
        print(f" DECISION-WINDOW ACCURACY SUMMARY - HELD-OUT: {sub_name}")
        print("="*88)
        print(f" {'Window':>6} | {'1s-Accum Sim':>13} | {'1s-Majority':>12} | {'5s-Accum Sim':>13} | {'Direct Eval':>12} | {'Decisions':>10}")
        print("-" * 88)
        for w in [1, 2, 5, 10, 15, 20, 25, 30, 35, 40]:
            r = multi_win_results[w]
            s_5s = f"{r['accum_5s']*100:11.2f}%" if r['accum_5s'] is not None else "        N/A"
            print(f" {w:4d} s | {r['accum_1s']*100:11.2f}% | {r['vote_1s']*100:10.2f}% | {s_5s} | {r['direct']*100:10.2f}% | {r['decisions']:10d}")
        print("="*88)
        
        # Also run canonical controls at 10s
        nc_zero, _ = evaluate_model(model, X_te_full, YA_te_full, YB_te_full, device, window_sec=10, zero_eeg=True, metric="pearson")
        nc_shuf, _ = evaluate_model(model, X_te_full, YA_te_full, YB_te_full, device, window_sec=10, shuffle_labels=True, metric="pearson")
        nc_swap, _ = evaluate_model(model, X_te_full, YA_te_full, YB_te_full, device, window_sec=10, swap_ab=True, metric="pearson")
        nt_10s = multi_win_results[10]["decisions"]
        print(f"  [Controls 10s] Zero EEG: {nc_zero/max(nt_10s,1)*100:.2f}% | Shuf Labels: {nc_shuf/max(nt_10s,1)*100:.2f}% | Swap A/B: {nc_swap/max(nt_10s,1)*100:.2f}%")
        
        # Fixed Lag Sweep (if requested)
        lag_sweep_res = None
        if sweep_lags:
            # Re-read envelopes for lag sweep
            _, env_sweep = get_mapping_data(audio_rep, audio_env_file)
            lag_sweep_res = run_lag_sweep(
                model, test_exs, fold_channels, lowcut, highcut, sub_name, mapping, env_sweep, 
                device, audio_layer_idx=audio_layer_idx, use_absolute_scoring=(loss_type == "absolute")
            )
            del env_sweep
            gc.collect()
            
        grand_summary[sub_name] = {
            "channels": fold_channels,
            "windows": multi_win_results,
            "lag_sweep": lag_sweep_res
        }
        
        # Save metrics per fold
        with open(Path(checkpoint_dir) / f"matchnet_fold_{sub_name}_metrics.json", "w") as f:
            json.dump(grand_summary[sub_name], f, indent=4)
            
        del X_te_full, YA_te_full, YB_te_full
        gc.collect()
        print(f"  Post-cleanup RAM: {psutil.virtual_memory().percent}% ({psutil.virtual_memory().used / 1e9:.2f} GB used)")

    # Print Grand Summary Table across all evaluated folds
    if len(grand_summary) > 1 or len(folds) == 18:
        print("\n" + "#"*92)
        print(" FULL DTU LEAVE-ONE-SUBJECT-OUT (LOSO) GRAND BENCHMARK TABLE")
        print("#"*92)
        print(f" {'Subject':>10} | {'5s (Accum)':>12} | {'10s (Accum)':>12} | {'15s (Accum)':>12} | {'20s (Accum)':>12} | {'30s (Accum)':>12} | {'40s (Accum)':>12}")
        print("-" * 92)
        
        win_keys = [5, 10, 15, 20, 30, 40]
        col_accs = {w: [] for w in win_keys}
        
        for sub, data in grand_summary.items():
            wins = data["windows"]
            row_str = f" {sub:>10} |"
            for w in win_keys:
                acc = wins[w]["accum_1s"] * 100
                col_accs[w].append(acc)
                row_str += f" {acc:10.2f}% |"
            print(row_str)
            
        print("-" * 92)
        mean_row = f" {'MEAN':>10} |"
        std_row = f" {'STD':>10} |"
        for w in win_keys:
            m = np.mean(col_accs[w])
            s = np.std(col_accs[w])
            mean_row += f" {m:10.2f}% |"
            std_row += f" {s:10.2f}% |"
        print(mean_row)
        print(std_row)
        print("#"*92)
        
        # Save grand summary
        with open(Path(checkpoint_dir) / "loso_grand_summary.json", "w") as f:
            json.dump(grand_summary, f, indent=4)
            print(f"Saved grand summary to {Path(checkpoint_dir) / 'loso_grand_summary.json'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate Contrastive MatchNet")
    parser.add_argument("--model", type=str, default="eegnet", choices=["eegnet", "atcnet", "eegnet_s1", "eegnet_s2", "eegnet_multiscale_m2", "sincalignnet", "msca"], help="Base EEG encoder")
    parser.add_argument("--channels", type=int, nargs='+', default=None, help="Explicit EEG channel indices to use")
    parser.add_argument("--num_channels", type=int, default=8, choices=[8, 16, 32, 64], help="Channel count for train-only ranking (8, 16, 32, 64)")
    parser.add_argument("--rank_channels", action="store_true", help="Rank channels using only training subjects per fold (zero test leakage)")
    parser.add_argument("--lowcut", type=float, default=1.0)
    parser.add_argument("--highcut", type=float, default=6.0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--subjects", type=str, nargs="+", help="Specific subjects to run (e.g., S1_data_preproc)")
    parser.add_argument("--loss", type=str, default="contrastive", choices=["contrastive", "anchored", "absolute", "dcca"], help="Loss function")
    parser.add_argument("--lambda_align", type=float, default=0.5, help="Weight for alignment penalty in anchored loss")
    parser.add_argument("--align_target", type=float, default=0.1, help="Positive alignment target for anchored loss")
    parser.add_argument("--augment_sign_flip", action="store_true", help="Randomly flip EEG sign during training")
    parser.add_argument("--use_dann", action="store_true", help="Use Domain Adversarial Neural Network")
    parser.add_argument("--use_temporal_transport", action="store_true", help="Enable Neural Temporal Deformation Field")
    parser.add_argument("--audio_rep", type=str, default="gammatone", choices=["gammatone", "wavlm"], help="Audio representation to use")
    parser.add_argument("--audio_env_file", type=str, default="", help="Path to audio features pkl file")
    parser.add_argument("--audio_layer_idx", type=int, default=1, help="WavLM layer index")
    parser.add_argument("--hard_negative_prob", type=float, default=0.0, help="Probability of temporal shifted negative")
    parser.add_argument("--lag_sec", type=float, default=0.0, help="Fixed temporal lag offset in seconds (e.g. 0.25 for +250ms)")
    parser.add_argument("--sweep_lags", action="store_true", help="Run a fixed lag sweep [-1.0s to +1.0s] during evaluation")
    parser.add_argument("--eval_only", action="store_true", help="Skip training and run evaluation on saved checkpoints")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save/load checkpoints")
    args = parser.parse_args()
    
    train_matchnet_loso(
        eeg_model=args.model,
        channels=args.channels,
        num_channels=args.num_channels,
        rank_channels=args.rank_channels,
        lowcut=args.lowcut,
        highcut=args.highcut,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        subjects_to_run=args.subjects,
        loss_type=args.loss,
        lambda_align=args.lambda_align,
        align_target=args.align_target,
        augment_sign_flip=args.augment_sign_flip,
        use_dann=args.use_dann,
        use_temporal_transport=args.use_temporal_transport,
        audio_rep=args.audio_rep,
        audio_env_file=args.audio_env_file,
        audio_layer_idx=args.audio_layer_idx,
        hard_negative_prob=args.hard_negative_prob,
        lag_sec=args.lag_sec,
        sweep_lags=args.sweep_lags,
        eval_only=args.eval_only,
        checkpoint_dir=args.checkpoint_dir
    )
