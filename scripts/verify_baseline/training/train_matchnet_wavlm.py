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

def prepare_dataset(examples, channels, lowcut, highcut, subject_id, mapping, envelopes, exclude_audio_files=None, audio_layer_idx=0):
    X = []
    Y_A = []
    Y_B = []
    
    sub_key = subject_id.replace("_data_preproc", "")
    
    for i, ex in enumerate(examples):
        trial_key = f"trial_{i}"
        
        if sub_key in mapping and trial_key in mapping[sub_key]:
            fname_a = mapping[sub_key][trial_key]["wavA"]["filename"]
            fname_b = mapping[sub_key][trial_key]["wavB"]["filename"]
            
            if exclude_audio_files is not None and (fname_a in exclude_audio_files or fname_b in exclude_audio_files):
                continue
            
            env_a = envelopes[fname_a]
            env_b = envelopes[fname_b]
            
            # CRITICAL FIX (DATASETS_REFERENCE.md): DTU convention is
            # wavA is ALWAYS the attended stream. wavB is ALWAYS unattended.
            # Event labels (1 or 2) indicate speaker gender, NOT attention.
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
            
        min_len = min(x_norm.shape[1], env_attended.shape[1])
        x_norm = x_norm[:, :min_len]
        env_attended = env_attended[:, :min_len]
        env_unattended = env_unattended[:, :min_len]
        
        env_attended = normalize_array(env_attended.T).T
        env_unattended = normalize_array(env_unattended.T).T
        
        X.append(x_norm.astype(np.float32))
        Y_A.append(env_attended.astype(np.float32))
        Y_B.append(env_unattended.astype(np.float32))
        
    return X, Y_A, Y_B

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
    printed_boundary = False
    
    np.random.seed(42)
    shuffle_indices = np.random.permutation(len(X))
    while np.any(shuffle_indices == np.arange(len(X))):
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
                
                if not printed_boundary:
                    print(f"--- WINDOW BOUNDARY CHECK ---")
                    print(f"Trial length (samples): {x_np.shape[1]}")
                    print(f"Eval start/end sample (EEG): {start} to {end}")
                    print(f"Eval start/end sample (Audio): {start} to {end}")
                    print(f"Window length: {window_samples} samples ({window_sec}s)")
                    printed_boundary = True
                    
                x_chunk = x_np[:, start:end].unsqueeze(0).to(device, dtype=torch.float32)
                
                if zero_eeg:
                    x_chunk = torch.zeros_like(x_chunk)
                    
                ya_chunk = ya_np[:, start:end].unsqueeze(0).to(device, dtype=torch.float32)
                yb_chunk = yb_np[:, start:end].unsqueeze(0).to(device, dtype=torch.float32)
                
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
        
        # Slicing a PyTorch tensor returns a view (zero memory allocation)
        x = self.X_full[trial_idx][:, start:end]
        ya = self.YA_full[trial_idx][:, start:end]
        yb = self.YB_full[trial_idx][:, start:end]
        
        if self.Subj_full is not None:
            subj = torch.tensor(self.Subj_full[trial_idx], dtype=torch.long)
            return x, ya, yb, subj
        return x, ya, yb

def train_matchnet_loso(eeg_model="eegnet", channels=[0, 33, 6, 41, 22, 59, 15, 52], lowcut=1.0, highcut=6.0, batch_size=128, num_workers=2, subjects_to_run=None, loss_type="contrastive", lambda_align=0.5, align_target=0.1, augment_sign_flip=False, use_dann=False, use_temporal_transport=False, audio_rep="gammatone", audio_env_file="", audio_layer_idx=0, hard_negative_prob=0.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prevent PyTorch multiprocessing memory duplication (Copy-on-Write failure) when using massive datasets
    if audio_rep == "wavlm":
        print("Forcing num_workers=0 for WavLM to prevent multiprocessing RAM explosion.")
        num_workers = 0
        
    print(f"Using device: {device} | MatchNet ({eeg_model}) | Channels: {channels}")
    
    all_paths = subject_files()
    if not all_paths:
        print("No subjects found.")
        return
        
    subject_examples = {str(p): load_subject_examples(p) for p in all_paths}
    folds = list(iter_leave_one_subject_out(all_paths))
    
    if subjects_to_run:
        folds = [f for f in folds if f[0].stem in subjects_to_run]
    
    os.makedirs(REPO_ROOT / "checkpoints", exist_ok=True)
    all_accs_norm_dict = {}
    all_accs_zero_dict = {}
    all_accs_shuf_dict = {}
    
    for held_out_path, train_paths in folds:
        held_out_key = str(held_out_path)
        print(f"\nEvaluating fold with held-out subject: {held_out_path.stem}")
        print(f"  [Memory] Pre-fold RAM: {psutil.virtual_memory().percent}% ({psutil.virtual_memory().used / 1e9:.2f} GB used)")
        
        # Load heavy audio features inside the loop so they can be deleted after extraction
        mapping, envelopes = get_mapping_data(audio_rep, audio_env_file)
        
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
            exs = subject_examples[str(p)]
            # Deep copy list to avoid in-place shuffling issues across folds
            exs = list(exs)
            
            np.random.seed(42)  # MUST be exactly 42 so all subjects split the exact same trials!
            np.random.shuffle(exs)
            
            # Exactly 10% of trials (e.g. 6 out of 60) go to validation
            val_split_num = int(0.1 * len(exs))
            if val_split_num == 0 and len(exs) > 0:
                val_split_num = 1
                
            val_exs = exs[:val_split_num]
            train_exs = exs[val_split_num:]
            
            # 2. Extract Training Trials
            tX, tYA, tYB = prepare_dataset(train_exs, channels, lowcut, highcut, p.stem, mapping, envelopes, audio_layer_idx=audio_layer_idx)
            X_tr_full.extend(tX)
            YA_tr_full.extend(tYA)
            YB_tr_full.extend(tYB)
            Subj_tr_full.extend([subj_id] * len(tX))
            
            # 3. Extract Validation Trials
            vX, vYA, vYB = prepare_dataset(val_exs, channels, lowcut, highcut, p.stem, mapping, envelopes, audio_layer_idx=audio_layer_idx)
            X_va_full.extend(vX)
            YA_va_full.extend(vYA)
            YB_va_full.extend(vYB)
            Subj_va_full.extend([subj_id] * len(vX))

        X_te_full, YA_te_full, YB_te_full = prepare_dataset(test_exs, channels, lowcut, highcut, held_out_path.stem, mapping, envelopes, audio_layer_idx=audio_layer_idx)
        
        # Free the massive envelopes dictionary now that everything is extracted
        del envelopes
        import gc
        gc.collect()
        
        # Convert all lists of numpy arrays to lists of PyTorch tensors to enable zero-copy slicing
        X_tr_full = [torch.from_numpy(x) for x in X_tr_full]
        YA_tr_full = [torch.from_numpy(x) for x in YA_tr_full]
        YB_tr_full = [torch.from_numpy(x) for x in YB_tr_full]
        
        X_va_full = [torch.from_numpy(x) for x in X_va_full]
        YA_va_full = [torch.from_numpy(x) for x in YA_va_full]
        YB_va_full = [torch.from_numpy(x) for x in YB_va_full]
        
        X_te_full = [torch.from_numpy(x) for x in X_te_full]
        YA_te_full = [torch.from_numpy(x) for x in YA_te_full]
        YB_te_full = [torch.from_numpy(x) for x in YB_te_full]
        
        # Chunk training data indices instead of copying arrays
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
        
        if len(X_va_full) == 0:
            raise RuntimeError("Validation split is empty. Check trial-level split before training.")
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=(num_workers > 0)
        )
            
        # Model
        num_subjects = len(train_paths)
        audio_channels = 768 if audio_rep == "wavlm" else 28
        model = ContrastiveMatchNet(
            eeg_model_type=eeg_model, 
            eeg_channels=len(channels), 
            audio_channels=audio_channels,
            latent_dim=64,
            use_temporal_transport=use_temporal_transport,
            audio_model_type=audio_rep
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scaler = torch.cuda.amp.GradScaler()
        
        best_val_acc = 0.0
        best_weights = deepcopy(model.state_dict())
        patience = 5
        epochs_no_improve = 0
        
        print(f"Training on {len(chunk_indices)} chunks ({TRAIN_WINDOW_SEC}s) | Batch Size: {batch_size} | Workers: {num_workers}...")
        
        for epoch in range(30):
            model.train()
            train_loss, train_sa, train_sb = 0.0, 0.0, 0.0
            train_loss_delay, train_loss_smooth, train_loss_mono, train_jac_min = 0.0, 0.0, 0.0, 0.0
            
            # Update GRL lambda for this epoch
            if use_dann:
                p = float(epoch) / 30.0
                grl_lambda = (2.0 / (1.0 + np.exp(-10.0 * p))) - 1.0
                model.grl.lambda_ = grl_lambda
                
            for batch in train_loader:
                if use_dann:
                    bx, bya, byb, b_subj = batch
                    b_subj = b_subj.to(device, non_blocking=True)
                else:
                    bx, bya, byb = batch
                    
                bx = bx.to(device, non_blocking=True)
                bya = bya.to(device, non_blocking=True)
                byb = byb.to(device, non_blocking=True)
                
                # Random Sign Flipping Augmentation
                if augment_sign_flip:
                    # Randomly multiply the entire EEG chunk by -1 (50% chance per batch element)
                    # Shape of bx: (B, C, T)
                    sign = torch.randint(0, 2, (bx.size(0), 1, 1), device=device).float() * 2.0 - 1.0
                    bx = bx * sign
                    
                # Hard Negative Sampling (Temporal Shift)
                # With hard_negative_prob probability, replace Y_B with a time-shifted Y_A
                if hard_negative_prob > 0.0 and torch.rand(1).item() < hard_negative_prob:
                    shift_amount = torch.randint(64, bya.size(-1) - 64, (1,)).item()
                    byb = torch.roll(bya, shifts=shift_amount, dims=-1)
                
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    if use_dann and use_temporal_transport:
                        z_eeg, z_a, z_b, subj_logits, (delta_t_a, delta_t_b) = model(bx, bya, byb, return_subject_logits=True, return_deltas=True)
                    elif use_dann:
                        z_eeg, z_a, z_b, subj_logits = model(bx, bya, byb, return_subject_logits=True)
                    elif use_temporal_transport:
                        z_eeg, z_a, z_b, (delta_t_a, delta_t_b) = model(bx, bya, byb, return_deltas=True)
                    else:
                        z_eeg, z_a, z_b = model(bx, bya, byb)
                        
                    if loss_type == "anchored":
                        loss, sa, sb = anchored_contrastive_loss(z_eeg, z_a, z_b, margin=0.1, lambda_align=lambda_align, align_target=align_target)
                    elif loss_type == "absolute":
                        # Absolute Magnitude Scoring for Ear-EEG phase invariance
                        sim_a = F.cosine_similarity(z_eeg, z_a, dim=1).mean(dim=1)
                        sim_b = F.cosine_similarity(z_eeg, z_b, dim=1).mean(dim=1)
                        loss = F.relu(0.1 - (torch.abs(sim_a) - torch.abs(sim_b))).mean()
                        sa = torch.abs(sim_a).mean()
                        sb = torch.abs(sim_b).mean()
                    elif loss_type == "dcca":
                        # DCCA operates on flattened [Batch * T, Features] tensors
                        B, D, T = z_eeg.shape
                        z_eeg_flat = z_eeg.transpose(1, 2).reshape(B * T, D)
                        z_a_flat = z_a.transpose(1, 2).reshape(B * T, D)
                        
                        loss, sa, sb = dcca_loss(z_eeg_flat, z_a_flat)
                    else:
                        loss, sa, sb = contrastive_loss(z_eeg, z_a, z_b, margin=0.1)
                        
                    if use_dann:
                        loss_subj = F.cross_entropy(subj_logits, b_subj)
                        loss = loss + loss_subj
                        
                    if use_temporal_transport:
                        # delta_t_a and delta_t_b are now in PHYSICAL SECONDS
                        loss_delay = (delta_t_a ** 2).mean() + (delta_t_b ** 2).mean()
                        
                        # Smoothness penalty
                        loss_smooth = ((delta_t_a[:, :, 1:] - delta_t_a[:, :, :-1]) ** 2).mean() + \
                                      ((delta_t_b[:, :, 1:] - delta_t_b[:, :, :-1]) ** 2).mean()
                                      
                        # Monotonicity penalty (Strategy 1.7)
                        dt = 1.0 / 64.0
                        mono_a = F.relu(-(dt + delta_t_a[:, :, 1:] - delta_t_a[:, :, :-1]))
                        mono_b = F.relu(-(dt + delta_t_b[:, :, 1:] - delta_t_b[:, :, :-1]))
                        loss_mono = (mono_a ** 2).mean() + (mono_b ** 2).mean()
                        
                        # Regularization weights for NTDF
                        loss = loss + (1.0 * loss_delay) + (10.0 * loss_smooth) + (100.0 * loss_mono)
                        
                        train_loss_delay += loss_delay.item()
                        train_loss_smooth += loss_smooth.item()
                        train_loss_mono += loss_mono.item()
                        r_t = 1.0 + 64.0 * (delta_t_a[:, :, 1:] - delta_t_a[:, :, :-1])
                        train_jac_min += r_t.min().item()
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
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
            avg_loss = train_loss / num_batches
            avg_sa = train_sa / num_batches
            avg_sb = train_sb / num_batches
            
            log_str = f"  Epoch {epoch+1:02d}/100 | Loss: {avg_loss:.4f} (sA: {avg_sa:.3f}, sB: {avg_sb:.3f}) | Val Acc: {val_acc*100:.2f}%"
            if use_temporal_transport:
                log_str += f" | NTDF[D:{train_loss_delay/num_batches:.4f} S:{train_loss_smooth/num_batches:.4f} M:{train_loss_mono/num_batches:.4f} Jac:{train_jac_min/num_batches:.3f}]"
            log_str += f" | Patience: {epochs_no_improve}/10"
            print(log_str)
                
            if epochs_no_improve >= patience:
                break
                
        # Checkpointing
        os.makedirs("checkpoints", exist_ok=True)
        final_path = f"checkpoints/matchnet_fold_{held_out_path.stem}_final.pth"
        torch.save(model.state_dict(), final_path)
        
        model.load_state_dict(best_weights)
        best_path = f"checkpoints/matchnet_fold_{held_out_path.stem}_best.pth"
        torch.save(best_weights, best_path)
        
        print(f"  [Evaluation - Pearson Correlation, 10s]")
        w_sec = 10
        
        nc_norm, nt_norm = evaluate_model(model, X_te_full, YA_te_full, YB_te_full, device, window_sec=w_sec, metric="pearson", use_absolute_scoring=(loss_type == "absolute"))
        acc_norm = nc_norm / max(nt_norm, 1)
        
        nc_zero, _ = evaluate_model(model, X_te_full, YA_te_full, YB_te_full, device, window_sec=w_sec, zero_eeg=True, metric="pearson", use_absolute_scoring=(loss_type == "absolute"))
        acc_zero = nc_zero / max(nt_norm, 1)
        
        nc_shuf, _ = evaluate_model(model, X_te_full, YA_te_full, YB_te_full, device, window_sec=w_sec, shuffle_labels=True, metric="pearson", use_absolute_scoring=(loss_type == "absolute"))
        acc_shuf = nc_shuf / max(nt_norm, 1)
        
        nc_shuf_eeg_time, _ = evaluate_model(model, X_te_full, YA_te_full, YB_te_full, device, window_sec=w_sec, shuffle_eeg_time=True, metric="pearson", use_absolute_scoring=(loss_type == "absolute"))
        acc_shuf_eeg_time = nc_shuf_eeg_time / max(nt_norm, 1)
        
        nc_shuf_audio_time, _ = evaluate_model(model, X_te_full, YA_te_full, YB_te_full, device, window_sec=w_sec, shuffle_audio_time=True, metric="pearson", use_absolute_scoring=(loss_type == "absolute"))
        acc_shuf_audio_time = nc_shuf_audio_time / max(nt_norm, 1)
        
        nc_perm_spatial, _ = evaluate_model(model, X_te_full, YA_te_full, YB_te_full, device, window_sec=w_sec, permute_spatial=True, metric="pearson", use_absolute_scoring=(loss_type == "absolute"))
        acc_perm_spatial = nc_perm_spatial / max(nt_norm, 1)
        
        nc_swap, _ = evaluate_model(model, X_te_full, YA_te_full, YB_te_full, device, window_sec=w_sec, swap_ab=True, metric="pearson", use_absolute_scoring=(loss_type == "absolute"))
        acc_swap = nc_swap / max(nt_norm, 1)
        
        print(f"    -> Window {w_sec:2d}s | Normal: {acc_norm*100:.2f}% | Decisions: {nt_norm}")
        print(f"    -> Controls   | Zero EEG: {acc_zero*100:.2f}% | Shuf Labels: {acc_shuf*100:.2f}%")
        print(f"    -> Controls   | Shuf EEG Time: {acc_shuf_eeg_time*100:.2f}% | Shuf Audio Time: {acc_shuf_audio_time*100:.2f}%")
        print(f"    -> Controls   | Permute Spatial: {acc_perm_spatial*100:.2f}% | Swap A/B: {acc_swap*100:.2f}%")
        
        if w_sec not in all_accs_norm_dict:
            all_accs_norm_dict[w_sec] = []
        
        all_accs_norm_dict[w_sec].append(acc_norm)
        
        fold_metrics = {
            "held_out": held_out_path.stem,
            "pearson": {
                "normal": {w: all_accs_norm_dict[w][-1] for w in all_accs_norm_dict}
            }
        }
        with open(f"checkpoints/matchnet_fold_{held_out_path.stem}_metrics.json", "w") as f:
            json.dump(fold_metrics, f, indent=4)
        
        # Aggressive memory cleanup to prevent swap death on Kaggle
        del X_tr_full, YA_tr_full, YB_tr_full, X_va_full, YA_va_full, YB_va_full, X_te_full, YA_te_full, YB_te_full
        gc.collect()
        
        print(f"  [Memory] Post-cleanup RAM: {psutil.virtual_memory().percent}% ({psutil.virtual_memory().used / 1e9:.2f} GB used)")
        
    print("\n" + "="*50)
    print(f"[MATCHNET ({eeg_model.upper()}) CANONICAL E0 EVALUATION (10s PEARSON)]")
    print("="*50)
    for w_sec in sorted(all_accs_norm_dict.keys()):
        final_acc_norm = np.mean(all_accs_norm_dict[w_sec])
        print(f" Window {w_sec:2d}s | Normal: {final_acc_norm*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Contrastive MatchNet")
    parser.add_argument("--model", type=str, default="eegnet", choices=["eegnet", "atcnet", "eegnet_s1", "eegnet_s2", "eegnet_multiscale_m2", "sincalignnet", "msca"], help="Base EEG encoder")
    parser.add_argument("--channels", type=int, nargs='+', default=[0, 33, 6, 41, 22, 59, 15, 52], help="EEG channel indices to use")
    parser.add_argument("--lowcut", type=float, default=1.0)
    parser.add_argument("--highcut", type=float, default=6.0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--subjects", type=str, nargs="+", help="Specific subjects to run (e.g., S1_data_preproc)")
    parser.add_argument("--loss", type=str, default="contrastive", choices=["contrastive", "anchored", "absolute", "dcca"], help="Loss function")
    parser.add_argument("--lambda_align", type=float, default=0.5, help="Weight for alignment penalty in anchored loss")
    parser.add_argument("--align_target", type=float, default=0.1, help="Positive alignment target for anchored loss")
    parser.add_argument("--augment_sign_flip", action="store_true", help="Randomly flip EEG sign during training to enforce phase-invariance")
    parser.add_argument("--use_dann", action="store_true", help="Use Domain Adversarial Neural Network to enforce subject invariance")
    parser.add_argument("--use_temporal_transport", action="store_true", help="Enable Neural Temporal Deformation Field (Strategy 1) to biologically warp audio delays")
    parser.add_argument("--audio_rep", type=str, default="gammatone", choices=["gammatone", "wavlm"], help="Audio representation to use")
    parser.add_argument("--audio_env_file", type=str, default="", help="Path to audio features pkl file (overrides default search)")
    parser.add_argument("--audio_layer_idx", type=int, default=1, help="Index of the WavLM layer to use (e.g. 0=L3, 1=L6, 2=L9, 3=L12)")
    parser.add_argument("--hard_negative_prob", type=float, default=0.0, help="Probability of using a temporal shifted attended stream as the negative")
    args = parser.parse_args()
    
    train_matchnet_loso(
        eeg_model=args.model,
        channels=args.channels,
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
        hard_negative_prob=args.hard_negative_prob
    )
