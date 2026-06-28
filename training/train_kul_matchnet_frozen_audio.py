import os
import sys
import re
import math
import numpy as np
import scipy.io
import argparse
import scipy.signal
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.matchnet import ContrastiveMatchNet, contrastive_loss
# We don't import from train_matchnet_loso.py to keep this clean, except maybe evaluating logic if needed,
# but we need Trial Accuracy logic which we will implement natively here for KUL.

FS = 64
TRAIN_WINDOW_SEC = 5
TRAIN_HOP_SEC = 2
DECISION_WINDOW_SEC = 10

def get_kul_subject_files():
    """Finds and sorts KUL subject files (S1 to S16)."""
    base_dirs = [
        Path("/kaggle/input/datasets/lowk1ee/s1-klu/"),
        Path("/kaggle/input/s1-klu/"),
        REPO_ROOT / "data" / "s1-klu",
        Path("data")
    ]
    
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
    # If it's a single trial, it might not be a numpy array
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
        
    from data.extract_gammatone_envelopes import extract_gammatone_envelopes
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

def chunk_data(x, ya, yb, window_sec, hop_sec, fs=FS):
    win_samples = int(window_sec * fs)
    hop_samples = int(hop_sec * fs)
    chunks_x, chunks_ya, chunks_yb = [], [], []
    start = 0
    while start + win_samples <= x.shape[1]:
        end = start + win_samples
        chunks_x.append(x[:, start:end])
        chunks_ya.append(ya[:, start:end])
        chunks_yb.append(yb[:, start:end])
        start += hop_samples
    return chunks_x, chunks_ya, chunks_yb

def evaluate_fold(model, test_data, device, window_sec=DECISION_WINDOW_SEC, fs=FS):
    """
    Evaluates a single subject (list of trials).
    Returns Window Accuracy, Trial Accuracy, Mean Margin, and Number of Trials.
    """
    model.eval()
    win_samples = int(window_sec * fs)
    
    total_windows = 0
    correct_windows = 0.0
    margins = []
    
    total_trials = len(test_data)
    correct_trials = 0.0
    total_trials_processed = 0
    
    with torch.no_grad():
        for x, ya, yb, meta in test_data:
            start = 0
            trial_sim_a = []
            trial_sim_b = []
            
            while start + win_samples <= x.shape[1]:
                end = start + win_samples
                cx = torch.FloatTensor(x[:, start:end]).unsqueeze(0).to(device)
                cya = torch.FloatTensor(ya[:, start:end]).unsqueeze(0).to(device)
                cyb = torch.FloatTensor(yb[:, start:end]).unsqueeze(0).to(device)
                
                z_eeg, z_a, z_b = model(cx, cya, cyb)
                sim_a = F.cosine_similarity(z_eeg, z_a, dim=1).mean().item()
                sim_b = F.cosine_similarity(z_eeg, z_b, dim=1).mean().item()
                
                if sim_a > sim_b: correct_windows += 1.0
                elif sim_a == sim_b: correct_windows += 0.5
                total_windows += 1
                
                margins.append(sim_a - sim_b)
                trial_sim_a.append(sim_a)
                trial_sim_b.append(sim_b)
                
                start += win_samples
                
            if trial_sim_a:
                mean_a = np.mean(trial_sim_a)
                mean_b = np.mean(trial_sim_b)
                margin = mean_a - mean_b
                
                pred = "CORRECT" if mean_a > mean_b else "WRONG" if mean_a < mean_b else "TIE"
                print(f"    Trial {meta['TrialID']:02d} | Exp: {meta['experiment']} | Track Attended: {meta['attended_track']} | Pred: {pred} | Margin: {margin:.4f}")
                
                if mean_a > mean_b: correct_trials += 1.0
                elif mean_a == mean_b: correct_trials += 0.5
                total_trials_processed += 1
                
    win_acc = correct_windows / max(total_windows, 1)
    trial_acc = correct_trials / max(total_trials_processed, 1)
    mean_margin = np.mean(margins) if margins else 0.0
    
    # DIAGNOSTICS: print summary for this fold
    print(f"  [EVAL SUMMARY] Total Trials Evaluated: {total_trials_processed}")
    print(f"  [EVAL SUMMARY] Correct: {correct_trials}, Accuracy: {trial_acc*100:.2f}%")
    
    return win_acc, trial_acc, mean_margin, total_trials_processed

def train_matchnet_kul_loso(dataset_dir=None, eeg_model_type="eegnet", epochs=50, batch_size=32, lr=1e-3, target_fold=None, dtu_ckpt=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting KUL LOSO Pipeline with Frozen Audio Encoder on {device}...")
    
    subject_paths = get_kul_subject_files()
    if not subject_paths:
        return
        
    print(f"Found {len(subject_paths)} subjects. Reading and preprocessing all data into RAM...")
    
    computed_envelope_cache = {}
    
    all_subject_data = {}
    for p in subject_paths:
        m = re.search(r"S(\d+)", p.name, re.IGNORECASE)
        sub_id = f"S{m.group(1)}"
        trials = load_kul_trials(str(p))
        
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
                    "attended_track": getattr(t, "attended_track", "Unknown")
                }
                valid_trials.append((x, ya, yb, meta))
            else:
                discard_reasons[reason] = discard_reasons.get(reason, 0) + 1
        print() # newline after trial progress
                
        all_subject_data[sub_id] = valid_trials
        
        print(f"Total trials in MAT file:      {len(trials)}")
        print(f"Trials after preprocessing:    {len(valid_trials)}")
        print(f"Trials discarded:              {len(trials) - len(valid_trials)}")
        if discard_reasons:
            print("\nReasons:")
            for reason, count in discard_reasons.items():
                print(f"  {reason}: {count}")
        print("-" * 40)
        
    os.makedirs(REPO_ROOT / "checkpoints", exist_ok=True)
    
    results = {}
    
    for held_out_idx, held_out_path in enumerate(subject_paths):
        m = re.search(r"S(\d+)", held_out_path.name, re.IGNORECASE)
        held_out_id = f"S{m.group(1)}"
        
        print(f"\n==================================================")
        print(f"Evaluating fold with held-out subject: {held_out_id}")
        print(f"==================================================")
        
        # Initialize model
        model = ContrastiveMatchNet("eegnet", eeg_channels=8, audio_channels=28, latent_dim=64).to(device)
        
        # Load DTU Pretrained Weights and Freeze Audio Encoder
        if dtu_ckpt is not None:
            print(f"Loading DTU pretrained weights from: {dtu_ckpt}")
            model.load_state_dict(torch.load(dtu_ckpt, map_location=device))
            
            # Freeze Audio Encoder
            for param in model.audio_encoder.parameters():
                param.requires_grad = False
            print("Successfully FROZE the Audio Encoder parameters.")
        else:
            print("WARNING: No DTU checkpoint provided. Audio encoder will NOT be frozen.")
        
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
        
        # Build Train and Val sets
        train_data_full = []
        for p in subject_paths:
            m2 = re.search(r"S(\d+)", p.name, re.IGNORECASE)
            sub_id = f"S{m2.group(1)}"
            if sub_id != held_out_id:
                train_data_full.extend(all_subject_data[sub_id])
                
        test_data = all_subject_data[held_out_id]
        
        # 10% Validation split from the training pool (trial level)
        np.random.seed(42)
        np.random.shuffle(train_data_full)
        val_split = int(0.1 * len(train_data_full))
        
        val_data = train_data_full[:val_split]
        train_data = train_data_full[val_split:]
        
        print(f"\n--- Phase D: Story-Balancing Training Data ---")
        track1_trials = [t for t in train_data if str(t[3].get('attended_track')) == '1']
        track2_trials = [t for t in train_data if str(t[3].get('attended_track')) == '2']
        
        print(f"Original Pool -> Track 1: {len(track1_trials)}, Track 2: {len(track2_trials)}")
        
        # Balance to the minority class
        min_class_size = min(len(track1_trials), len(track2_trials))
        
        # Randomly sample the majority class
        np.random.shuffle(track1_trials)
        np.random.shuffle(track2_trials)
        
        balanced_train_data = track1_trials[:min_class_size] + track2_trials[:min_class_size]
        np.random.shuffle(balanced_train_data)
        
        print(f"Balanced Pool -> Track 1: {min_class_size}, Track 2: {min_class_size}")
        print(f"Total balanced training trials: {len(balanced_train_data)}")
        print(f"----------------------------------------------\n")
        
        # Chunk training data
        tr_x, tr_ya, tr_yb = [], [], []
        for x, ya, yb, _ in balanced_train_data:
            cx, cya, cyb = chunk_data(x, ya, yb, TRAIN_WINDOW_SEC, TRAIN_HOP_SEC)
            tr_x.extend(cx); tr_ya.extend(cya); tr_yb.extend(cyb)
            
        print(f"Training on {len(tr_x)} chunks | Validating on {len(val_data)} full trials...")
        
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(np.stack(tr_x)), 
                          torch.FloatTensor(np.stack(tr_ya)), 
                          torch.FloatTensor(np.stack(tr_yb))),
            batch_size=128, shuffle=True, pin_memory=True
        )
        
        model = ContrastiveMatchNet("eegnet", eeg_channels=8, audio_channels=28, latent_dim=64).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler('cuda')
            use_amp = True
        else:
            scaler = torch.cuda.amp.GradScaler()
            use_amp = False
            
        best_val_acc = 0.0
        best_weights = deepcopy(model.state_dict())
        patience = 5
        epochs_no_improve = 0
        
        for epoch in range(100):
            model.train()
            train_loss = 0.0
            for bx, bya, byb in train_loader:
                bx, bya, byb = bx.to(device, non_blocking=True), bya.to(device, non_blocking=True), byb.to(device, non_blocking=True)
                optimizer.zero_grad()
                
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        z_eeg, z_a, z_b = model(bx, bya, byb)
                        loss, _, _ = contrastive_loss(z_eeg, z_a, z_b, margin=0.1)
                else:
                    with torch.cuda.amp.autocast():
                        z_eeg, z_a, z_b = model(bx, bya, byb)
                        loss, _, _ = contrastive_loss(z_eeg, z_a, z_b, margin=0.1)
                        
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()
                
            win_acc, _, _, _ = evaluate_fold(model, val_data, device, window_sec=DECISION_WINDOW_SEC)
            
            # Use carriage return to print smoothly like tqdm
            sys.stdout.write(f"\\r  Epoch {epoch+1:02d} | Train Loss: {train_loss/len(train_loader):.4f} | Val Window Acc: {win_acc*100:.2f}% | No Improve: {epochs_no_improve}")
            sys.stdout.flush()
            
            if win_acc > best_val_acc:
                best_val_acc = win_acc
                best_weights = deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break
        print()
        
        model.load_state_dict(best_weights)
        save_path = os.path.join("checkpoints", f"matchnet_kul_frozen_fold_{held_out_id}_best.pth")
        torch.save(model.state_dict(), save_path)
        
        win_acc, trial_acc, mean_margin, num_trials = evaluate_fold(model, test_data, device, window_sec=DECISION_WINDOW_SEC)
        
        results[held_out_id] = {
            "win_acc": win_acc,
            "trial_acc": trial_acc,
            "margin": mean_margin,
            "trials": num_trials
        }
        
        print(f"Fold {held_out_id} Results -> Window Acc: {win_acc*100:.2f}% | Trial Acc: {trial_acc*100:.2f}% | Margin: {mean_margin:.4f}")
        
    print("\n==================================================")
    print("KUL LOSO RESULTS")
    print("==================================================")
    print(f"{'Subject':<12} {'WindowAcc':<13} {'TrialAcc':<12} {'Margin':<10}")
    
    mean_w = []
    mean_t = []
    
    for sub_id in sorted(results.keys(), key=lambda x: int(x[1:])):
        r = results[sub_id]
        print(f"{sub_id:<12} {r['win_acc']*100:>8.2f}% {r['trial_acc']*100:>10.2f}% {r['margin']:>11.4f}")
        mean_w.append(r['win_acc'])
        mean_t.append(r['trial_acc'])
        
    print("-" * 46)
    print(f"Mean Window Accuracy : {np.mean(mean_w)*100:.2f}%")
    print(f"Mean Trial Accuracy  : {np.mean(mean_t)*100:.2f}%")
    print(f"Std Trial Accuracy   : {np.std(mean_t)*100:.2f}%")
    print("==================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default=None, help="Path to Kaggle dataset if applicable")
    parser.add_argument("--eeg_model", type=str, default="eegnet", choices=["eegnet", "atcnet", "eegnet_tcn"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--fold", type=str, default=None, help="Specific subject to use as held-out test set (e.g. 'S1'). If None, trains all folds.")
    parser.add_argument("--dtu_ckpt", type=str, default=None, help="Path to DTU pretrained model to freeze audio encoder")
    args = parser.parse_args()
    
    if args.dtu_ckpt is None:
        print("ERROR: --dtu_ckpt is required for the Phase E intervention!")
        sys.exit(1)
        
    train_matchnet_kul_loso(args.dataset_dir, args.eeg_model, args.epochs, args.batch_size, args.lr, args.fold, args.dtu_ckpt)
