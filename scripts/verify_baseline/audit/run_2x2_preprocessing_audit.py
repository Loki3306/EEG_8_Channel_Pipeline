import argparse
import sys
import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from scipy import signal
from copy import deepcopy

REPO_ROOT = Path(__file__).resolve().parents[3]
VERIFY_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(VERIFY_ROOT) not in sys.path:
    sys.path.insert(0, str(VERIFY_ROOT))

from models.catcn import CATCNDirectDecoder
from src.streaming.causal_filters import StreamingCausalEEGFilter
from baselines.ridge_aad import load_subject_examples, subject_files
from training.train_matchnet_wavlm import FS, TRAIN_WINDOW_SEC, prepare_dataset, get_mapping_data
from training.montages import MONTAGES, DTU_CHANNELS

def butter_bandpass_filtfilt(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 2) -> np.ndarray:
    """Zero-phase offline filter."""
    b, a = signal.butter(order, [lowcut, highcut], btype='bandpass', fs=fs)
    return signal.filtfilt(b, a, data, axis=0)

def butter_lowpass_sosfilt(data: np.ndarray, cutoff: float, fs: float, order: int = 2) -> np.ndarray:
    """Causal low-pass filter for audio envelopes."""
    sos = signal.butter(order, cutoff, btype='low', fs=fs, output='sos')
    zi = signal.sosfilt_zi(sos) * (data[0] if len(data) > 0 else 0.0)
    out, _ = signal.sosfilt(sos, data, zi=zi)
    return out

def evaluate_condition(model, eeg_list, ya_list, yb_list, window_samples, device):
    """Evaluates non-overlapping windows across a set of trials."""
    deltas = []
    for eeg, ya, yb in zip(eeg_list, ya_list, yb_list):
        t_len = len(eeg)
        for s in range(0, t_len - window_samples + 1, window_samples):
            e = s + window_samples
            w_e = torch.from_numpy(eeg[s:e].T.copy()).unsqueeze(0).float().to(device)
            w_a = torch.from_numpy(ya[s:e].copy()).unsqueeze(0).unsqueeze(0).float().to(device)
            w_b = torch.from_numpy(yb[s:e].copy()).unsqueeze(0).unsqueeze(0).float().to(device)
            with torch.no_grad():
                d, _, _ = model(w_e, w_a, w_b)
            deltas.append(d.item())
            
    deltas = np.array(deltas)
    acc = np.mean(deltas > 0) * 100.0 if len(deltas) > 0 else 50.0
    return acc, deltas

def run_2x2_audit(args):
    fs = FS
    montage_channels = MONTAGES[args.montage]
    n_ch = len(montage_channels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 90)
    print("  2x2 FACTORIAL PREPROCESSING ABLATION AUDIT")
    print(f"  Target: {args.subject} | Montage: {args.montage} ({n_ch} channels) | Device: {device}")
    print("=" * 90)
    
    # 1. Subject Files
    all_paths = subject_files()
    held_out_path = None
    train_paths = []
    for p in all_paths:
        if p.stem == args.subject:
            held_out_path = p
        else:
            train_paths.append(p)
            
    is_real_data = held_out_path is not None
    
    # 2. Setup Model
    torch.manual_seed(42)
    np.random.seed(42)
    model = CATCNDirectDecoder(eeg_channels=n_ch, audio_channels=1, hidden_dim=64, max_lag_samples=8).to(device)
    
    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"[MODEL] Loading trained checkpoint from: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state.get("model_state_dict", state.get("state_dict", state)))
        print("[MODEL] Checkpoint loaded successfully!")
    elif is_real_data and args.train_epochs > 0:
        mode_str = "CAUSAL STREAMING (Matched)" if args.train_causal else "OFFLINE ZERO-PHASE"
        print(f"\n[MODEL TRAINING]: Training CA-TCN ({mode_str}) for {args.train_epochs} epoch(s) on {len(train_paths)} subjects...")
        mapping, envelopes = get_mapping_data("gammatone")
        win_samples = int(TRAIN_WINDOW_SEC * fs)
        hop_samples = int(2.5 * fs)
        
        train_causal_filter = StreamingCausalEEGFilter(lowcut=1.0, highcut=6.0, fs=fs, order=2, n_channels=n_ch) if args.train_causal else None
        
        X_tr_list, YA_tr_list, YB_tr_list = [], [], []
        for p in train_paths:
            sub_name = p.stem
            exs = list(load_subject_examples(p))
            
            if args.train_causal:
                _, YA_sub_raw, YB_sub_raw = prepare_dataset(exs, montage_channels, 1.0, 6.0, sub_name, mapping, envelopes)
                YA_sub = [ya.mean(axis=0, keepdims=True).squeeze(0).astype(np.float32) if ya.shape[0] > 1 else ya.squeeze(0).astype(np.float32) for ya in YA_sub_raw]
                YB_sub = [yb.mean(axis=0, keepdims=True).squeeze(0).astype(np.float32) if yb.shape[0] > 1 else yb.squeeze(0).astype(np.float32) for yb in YB_sub_raw]
                
                for idx in range(min(len(exs), len(YA_sub))):
                    raw_eeg = exs[idx].eeg[:, montage_channels].astype(np.float32)
                    min_len = min(len(raw_eeg), len(YA_sub[idx]), len(YB_sub[idx]))
                    raw_eeg = raw_eeg[:min_len]
                    
                    train_causal_filter.reset()
                    eeg_c = train_causal_filter.process_chunk(raw_eeg)
                    eeg_c = (eeg_c - np.mean(eeg_c, axis=0, keepdims=True)) / (np.std(eeg_c, axis=0, keepdims=True) + 1e-12)
                    
                    ya_c = butter_lowpass_sosfilt(YA_sub[idx][:min_len], 8.0, fs, order=2).astype(np.float32)
                    yb_c = butter_lowpass_sosfilt(YB_sub[idx][:min_len], 8.0, fs, order=2).astype(np.float32)
                    ya_c = (ya_c - np.mean(ya_c)) / (np.std(ya_c) + 1e-12)
                    yb_c = (yb_c - np.mean(yb_c)) / (np.std(yb_c) + 1e-12)
                    
                    x_t = eeg_c.T
                    ya_t = np.expand_dims(ya_c, axis=0)
                    yb_t = np.expand_dims(yb_c, axis=0)
                    
                    t_len = min_len
                    start = 0
                    while start + win_samples <= t_len:
                        end = start + win_samples
                        X_tr_list.append(x_t[:, start:end])
                        YA_tr_list.append(ya_t[:, start:end])
                        YB_tr_list.append(yb_t[:, start:end])
                        start += hop_samples
            else:
                X_sub, YA_sub, YB_sub = prepare_dataset(exs, montage_channels, 1.0, 6.0, sub_name, mapping, envelopes)
                YA_sub = [ya.mean(axis=0, keepdims=True).astype(np.float32) if ya.shape[0] > 1 else ya.astype(np.float32) for ya in YA_sub]
                YB_sub = [yb.mean(axis=0, keepdims=True).astype(np.float32) if yb.shape[0] > 1 else yb.astype(np.float32) for yb in YB_sub]
                
                for idx in range(len(X_sub)):
                    x, ya, yb = X_sub[idx], YA_sub[idx], YB_sub[idx]
                    t_len = x.shape[1]
                    start = 0
                    while start + win_samples <= t_len:
                        end = start + win_samples
                        X_tr_list.append(x[:, start:end])
                        YA_tr_list.append(ya[:, start:end])
                        YB_tr_list.append(yb[:, start:end])
                        start += hop_samples
                    
        train_ds = TensorDataset(
            torch.from_numpy(np.stack(X_tr_list, axis=0)),
            torch.from_numpy(np.stack(YA_tr_list, axis=0)),
            torch.from_numpy(np.stack(YB_tr_list, axis=0))
        )
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, pin_memory=True)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        model.train()
        for epoch in range(args.train_epochs):
            total_loss = 0.0
            for bx, bya, byb in train_loader:
                bx, bya, byb = bx.to(device), bya.to(device), byb.to(device)
                optimizer.zero_grad()
                delta, (la, lb), _ = model(bx, bya, byb)
                loss = torch.clamp(0.5 - (la - lb), min=0.0).mean()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"  * Epoch {epoch+1}/{args.train_epochs} Loss: {total_loss/len(train_loader):.4f}")
        print("[MODEL TRAINING] Complete. Proceeding to 2x2 Factorial Audit.")
        save_path = Path("/kaggle/working/catcn_deployment_weights.pt")
        try:
            torch.save(model.state_dict(), save_path)
            print(f"[MODEL TRAINING] Saved trained weights to: {save_path}")
        except Exception as e:
            pass
    else:
        print("[MODEL WARNING] Running without trained weights. Provide --checkpoint or set --train_epochs > 0 for meaningful accuracy.")
        
    model.eval()
    
    # 3. Load Real Held-out Subject Data
    if not is_real_data:
        print("[ERROR] Real DTU patient files not found. The 2x2 factorial audit requires real EEG recordings.")
        return
        
    print(f"\n[DATA]: Loading genuine DTU patient recording for {held_out_path.name}...")
    mapping, envelopes = get_mapping_data("gammatone")
    test_exs = list(load_subject_examples(held_out_path))
    ch_indices = montage_channels
    
    # Prepare offline reference audio envelopes
    _, YA_offline_raw, YB_offline_raw = prepare_dataset(test_exs, montage_channels, 1.0, 6.0, args.subject, mapping, envelopes)
    YA_offline = [ya.mean(axis=0, keepdims=True).squeeze(0).astype(np.float32) if ya.shape[0] > 1 else ya.squeeze(0).astype(np.float32) for ya in YA_offline_raw]
    YB_offline = [yb.mean(axis=0, keepdims=True).squeeze(0).astype(np.float32) if yb.shape[0] > 1 else yb.squeeze(0).astype(np.float32) for yb in YB_offline_raw]
    
    raw_eeg_list = []
    eeg_offline_list = []
    eeg_causal_list = []
    audio_causal_a_list = []
    audio_causal_b_list = []
    
    causal_eeg_filter = StreamingCausalEEGFilter(lowcut=1.0, highcut=6.0, fs=fs, order=2, n_channels=n_ch)
    
    n_eval_trials = min(args.trials, len(test_exs), len(YA_offline))
    for idx in range(n_eval_trials):
        raw_eeg = test_exs[idx].eeg[:, ch_indices].astype(np.float32) # [T, C]
        min_len = min(len(raw_eeg), len(YA_offline[idx]), len(YB_offline[idx]))
        raw_eeg = raw_eeg[:min_len]
        ya_off = YA_offline[idx][:min_len]
        yb_off = YB_offline[idx][:min_len]
        
        # 1. Offline EEG: zero-phase filtfilt + per-channel standardization
        eeg_off = butter_bandpass_filtfilt(raw_eeg, 1.0, 6.0, fs, order=2)
        eeg_off = (eeg_off - np.mean(eeg_off, axis=0, keepdims=True)) / (np.std(eeg_off, axis=0, keepdims=True) + 1e-12)
        eeg_offline_list.append(eeg_off)
        
        # 2. Causal EEG: streaming sosfilt + per-channel standardization
        causal_eeg_filter.reset()
        eeg_caus = causal_eeg_filter.process_chunk(raw_eeg)
        eeg_caus = (eeg_caus - np.mean(eeg_caus, axis=0, keepdims=True)) / (np.std(eeg_caus, axis=0, keepdims=True) + 1e-12)
        eeg_causal_list.append(eeg_caus)
        
        # 3. Causal Audio: causal lowpass filter on rectified envelope
        ya_caus = butter_lowpass_sosfilt(ya_off, 8.0, fs, order=2).astype(np.float32)
        yb_caus = butter_lowpass_sosfilt(yb_off, 8.0, fs, order=2).astype(np.float32)
        ya_caus = (ya_caus - np.mean(ya_caus)) / (np.std(ya_caus) + 1e-12)
        yb_caus = (yb_caus - np.mean(yb_caus)) / (np.std(yb_caus) + 1e-12)
        audio_causal_a_list.append(ya_caus)
        audio_causal_b_list.append(yb_caus)
        
        # Standardize offline audio
        YA_offline[idx] = (ya_off - np.mean(ya_off)) / (np.std(ya_off) + 1e-12)
        YB_offline[idx] = (yb_off - np.mean(yb_off)) / (np.std(yb_off) + 1e-12)
        
    print(f"[DATA] Prepared {n_eval_trials} trials across all 4 preprocessing conditions.")
    
    # -------------------------------------------------------------
    # 4. Evaluate 2x2 Factorial Matrix for Windows W in {5s, 10s}
    # -------------------------------------------------------------
    for w_sec in [5.0, 10.0]:
        w_samp = int(w_sec * fs)
        print("\n" + "=" * 90)
        print(f"  2x2 MATRIX EVALUATION — DECISION WINDOW: {w_sec:.1f} s ({w_samp} samples)")
        print("=" * 90)
        
        # Cell A: Offline EEG + Offline Audio (Reference)
        acc_A, d_A = evaluate_condition(model, eeg_offline_list, YA_offline[:n_eval_trials], YB_offline[:n_eval_trials], w_samp, device)
        
        # Cell B: Offline EEG + Causal Audio
        acc_B, d_B = evaluate_condition(model, eeg_offline_list, audio_causal_a_list, audio_causal_b_list, w_samp, device)
        
        # Cell C: Causal EEG + Offline Audio
        acc_C, d_C = evaluate_condition(model, eeg_causal_list, YA_offline[:n_eval_trials], YB_offline[:n_eval_trials], w_samp, device)
        
        # Cell D: Causal EEG + Causal Audio (Full Deployment)
        acc_D, d_D = evaluate_condition(model, eeg_causal_list, audio_causal_a_list, audio_causal_b_list, w_samp, device)
        
        cost_audio = acc_B - acc_A
        cost_eeg = acc_C - acc_A
        cost_total = acc_D - acc_A
        
        print(f"\n  {'Preprocessing':<18} | {'Audio: Offline':<18} | {'Audio: Causal Streaming'}")
        print("  " + "-" * 65)
        print(f"  {'EEG: Offline':<18} | {acc_A:>12.2f}% (A)   | {acc_B:>12.2f}% (B)")
        print(f"  {'EEG: Causal':<18}  | {acc_C:>12.2f}% (C)   | {acc_D:>12.2f}% (D)")
        print("  " + "-" * 65)
        
        print("\n  FACTORIAL ATTRIBUTION BREAKDOWN:")
        print(f"  * [A] Baseline Offline Performance:    {acc_A:.2f}%")
        print(f"  * Audio Streaming Impact (B - A):      {cost_audio:+.2f} pp")
        print(f"  * EEG Streaming Impact   (C - A):      {cost_eeg:+.2f} pp")
        print(f"  * Total Deployment Gap   (D - A):      {cost_total:+.2f} pp")
        print(f"  * [D] Full Causal Streaming Accuracy:  {acc_D:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2x2 Factorial Preprocessing Audit")
    parser.add_argument("--subject", type=str, default="S1_data_preproc", help="Subject name")
    parser.add_argument("--montage", type=str, default="near_ear_expanded", help="Montage name")
    parser.add_argument("--trials", type=int, default=20, help="Number of trials to evaluate")
    parser.add_argument("--train_epochs", type=int, default=3, help="Training epochs to train on training fold if no checkpoint provided (default: 3)")
    parser.add_argument("--train_causal", action="store_true", help="Train CA-TCN directly on causal-filtered streaming EEG and audio to eliminate the phase mismatch gap")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to pre-trained model checkpoint")
    args = parser.parse_args()
    
    run_2x2_audit(args)
