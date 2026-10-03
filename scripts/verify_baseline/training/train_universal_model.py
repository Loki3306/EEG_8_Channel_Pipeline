import argparse
import sys
import os
import json
import time
import math
from pathlib import Path
from copy import deepcopy
from datetime import datetime
import numpy as np
from scipy import signal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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

def butter_lowpass_sosfilt(data: np.ndarray, cutoff: float, fs: float, order: int = 2) -> np.ndarray:
    """Causal low-pass filter for audio envelopes."""
    sos = signal.butter(order, cutoff, btype='low', fs=fs, output='sos')
    zi = signal.sosfilt_zi(sos) * (data[0] if len(data) > 0 else 0.0)
    out, _ = signal.sosfilt(sos, data, zi=zi)
    return out

def evaluate_windows(model, eeg_list, ya_list, yb_list, window_samples, device):
    """Evaluates non-overlapping windows across a set of trials."""
    deltas = []
    model.eval()
    with torch.no_grad():
        for eeg, ya, yb in zip(eeg_list, ya_list, yb_list):
            t_len = min(len(eeg), len(ya), len(yb))
            for s in range(0, t_len - window_samples + 1, window_samples):
                e = s + window_samples
                w_e = torch.from_numpy(eeg[s:e].T.copy()).unsqueeze(0).float().to(device)
                w_a = torch.from_numpy(ya[s:e].copy()).unsqueeze(0).unsqueeze(0).float().to(device)
                w_b = torch.from_numpy(yb[s:e].copy()).unsqueeze(0).unsqueeze(0).float().to(device)
                d, _, _ = model(w_e, w_a, w_b)
                deltas.append(d.item())
                
    deltas = np.array(deltas)
    if len(deltas) == 0:
        return 50.0, 50.0, 0
    std_acc = float(np.mean(deltas > 0) * 100.0)
    inv_acc = float(np.mean(deltas < 0) * 100.0)
    pol_acc = max(std_acc, inv_acc)
    return std_acc, pol_acc, len(deltas)

def run_universal_training(args):
    fs = FS
    montage_channels = MONTAGES[args.montage]
    n_ch = len(montage_channels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 96)
    print("  UNIVERSAL FOUNDATION MODEL TRAINING (ALL DTU SUBJECTS)")
    print(f"  Montage: {args.montage} ({n_ch} channels) | Preprocessing: CAUSAL STREAMING (Matched)")
    print(f"  Device: {device} | Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print("=" * 96)
    
    # 1. Discover all DTU subjects
    all_paths = subject_files()
    if not all_paths or len(all_paths) < 18:
        if Path("/kaggle/input").exists():
            rglobbed = list(Path("/kaggle/input").rglob("S*_data_preproc.mat"))
            if len(rglobbed) > len(all_paths):
                by_stem = {p.stem: p for p in rglobbed}
                all_paths = sorted(by_stem.values(), key=lambda path: int(path.stem.split("_")[0][1:]))
    if not all_paths:
        raise FileNotFoundError("No DTU patient files found. Please mount the DTU dataset in /kaggle/input.")
    print(f"[DATA] Discovered {len(all_paths)} DTU subjects: {[p.stem for p in all_paths]}")
    
    mapping, envelopes = get_mapping_data("gammatone")
    win_samples = int(args.window_sec * fs)
    hop_samples = int(args.hop_sec * fs)
    causal_eeg_filter = StreamingCausalEEGFilter(lowcut=1.0, highcut=6.0, fs=fs, order=2, n_channels=n_ch)
    
    # 2. Extract and Preprocess All Subjects
    print("\n[DATA PREPARATION]: Extracting causal streaming EEG and speech envelopes across all subjects...")
    t_data_start = time.time()
    
    subject_train_data = {}
    subject_test_data = {}
    
    total_train_trials = 0
    total_test_trials = 0
    
    X_tr_list, YA_tr_list, YB_tr_list = [], [], []
    X_va_list, YA_va_list, YB_va_list = [], [], []
    
    for p in all_paths:
        sub_name = p.stem
        exs = list(load_subject_examples(p))
        
        # Prepare offline reference envelopes
        _, YA_raw, YB_raw = prepare_dataset(exs, montage_channels, 1.0, 6.0, sub_name, mapping, envelopes)
        YA_clean = [ya.mean(axis=0, keepdims=True).squeeze(0).astype(np.float32) if ya.shape[0] > 1 else ya.squeeze(0).astype(np.float32) for ya in YA_raw]
        YB_clean = [yb.mean(axis=0, keepdims=True).squeeze(0).astype(np.float32) if yb.shape[0] > 1 else yb.squeeze(0).astype(np.float32) for yb in YB_raw]
        
        n_valid = min(len(exs), len(YA_clean))
        if n_valid == 0:
            continue
            
        # Split: First 80% train, last 20% held-out test
        split_idx = int(math.floor(n_valid * (1.0 - args.test_split)))
        
        sub_eeg_tr, sub_ya_tr, sub_yb_tr = [], [], []
        sub_eeg_te, sub_ya_te, sub_yb_te = [], [], []
        
        for idx in range(n_valid):
            raw_eeg = exs[idx].eeg[:, montage_channels].astype(np.float32)
            min_len = min(len(raw_eeg), len(YA_clean[idx]), len(YB_clean[idx]))
            raw_eeg = raw_eeg[:min_len]
            
            # Causal EEG filtering + standardization
            causal_eeg_filter.reset()
            eeg_c = causal_eeg_filter.process_chunk(raw_eeg)
            eeg_c = (eeg_c - np.mean(eeg_c, axis=0, keepdims=True)) / (np.std(eeg_c, axis=0, keepdims=True) + 1e-12)
            
            # Causal Audio lowpass + standardization
            ya_c = butter_lowpass_sosfilt(YA_clean[idx][:min_len], 8.0, fs, order=2).astype(np.float32)
            yb_c = butter_lowpass_sosfilt(YB_clean[idx][:min_len], 8.0, fs, order=2).astype(np.float32)
            ya_c = (ya_c - np.mean(ya_c)) / (np.std(ya_c) + 1e-12)
            yb_c = (yb_c - np.mean(yb_c)) / (np.std(yb_c) + 1e-12)
            
            if idx < split_idx:
                sub_eeg_tr.append(eeg_c)
                sub_ya_tr.append(ya_c)
                sub_yb_tr.append(yb_c)
                total_train_trials += 1
                
                # Chunk training trials for DataLoader
                x_t = eeg_c.T
                ya_t = np.expand_dims(ya_c, axis=0)
                yb_t = np.expand_dims(yb_c, axis=0)
                
                # Assign 90% chunks to train, 10% to validation
                is_val_trial = (idx % 10 == 0)
                start = 0
                while start + win_samples <= min_len:
                    end = start + win_samples
                    if is_val_trial:
                        X_va_list.append(x_t[:, start:end])
                        YA_va_list.append(ya_t[:, start:end])
                        YB_va_list.append(yb_t[:, start:end])
                    else:
                        X_tr_list.append(x_t[:, start:end])
                        YA_tr_list.append(ya_t[:, start:end])
                        YB_tr_list.append(yb_t[:, start:end])
                    start += hop_samples
            else:
                sub_eeg_te.append(eeg_c)
                sub_ya_te.append(ya_c)
                sub_yb_te.append(yb_c)
                total_test_trials += 1
                
        subject_test_data[sub_name] = (sub_eeg_te, sub_ya_te, sub_yb_te)
        print(f"  * {sub_name:<16}: {len(sub_eeg_tr)} train trials, {len(sub_eeg_te)} held-out test trials")
        
    print(f"[DATA] Prepared {total_train_trials} train trials ({len(X_tr_list)} chunks) & {total_test_trials} test trials in {time.time() - t_data_start:.1f}s.")
    
    # 3. Create PyTorch Datasets
    train_ds = TensorDataset(
        torch.from_numpy(np.stack(X_tr_list, axis=0)),
        torch.from_numpy(np.stack(YA_tr_list, axis=0)),
        torch.from_numpy(np.stack(YB_tr_list, axis=0))
    )
    val_ds = TensorDataset(
        torch.from_numpy(np.stack(X_va_list, axis=0)),
        torch.from_numpy(np.stack(YA_va_list, axis=0)),
        torch.from_numpy(np.stack(YB_va_list, axis=0))
    )
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2 if os.name != 'nt' else 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    
    # 4. Initialize Universal Model
    torch.manual_seed(42)
    np.random.seed(42)
    model = CATCNDirectDecoder(eeg_channels=n_ch, audio_channels=1, hidden_dim=args.hidden_dim, max_lag_samples=8).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
    
    best_val_loss = float('inf')
    best_weights = deepcopy(model.state_dict())
    
    # 5. Training Loop
    print("\n" + "=" * 96)
    print(f"  COMMENCING UNIVERSAL FOUNDATION TRAINING ({args.epochs} EPOCHS)")
    print("=" * 96)
    
    train_start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        t_epoch_start = time.time()
        model.train()
        train_loss = 0.0
        n_train_batches = 0
        
        for bx, bya, byb in train_loader:
            bx = bx.to(device, non_blocking=True)
            bya = bya.to(device, non_blocking=True)
            byb = byb.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                delta, (la, lb), _ = model(bx, bya, byb)
                # Margin Ranking Loss: require la > lb + 0.5
                loss = torch.clamp(0.5 - (la - lb), min=0.0).mean()
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            n_train_batches += 1
            
        scheduler.step()
        avg_train_loss = train_loss / max(1, n_train_batches)
        
        # Validation Evaluation
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for bx, bya, byb in val_loader:
                bx = bx.to(device, non_blocking=True)
                bya = bya.to(device, non_blocking=True)
                byb = byb.to(device, non_blocking=True)
                with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                    delta, (la, lb), _ = model(bx, bya, byb)
                    v_loss = torch.clamp(0.5 - (la - lb), min=0.0).mean()
                val_loss += v_loss.item()
                n_val_batches += 1
                
        avg_val_loss = val_loss / max(1, n_val_batches)
        epoch_sec = time.time() - t_epoch_start
        
        # Checkpointing
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            best_weights = deepcopy(model.state_dict())
            star_flag = " [*BEST*]"
        else:
            star_flag = ""
            
        print(f"  Epoch [{epoch:02d}/{args.epochs:02d}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | Time: {epoch_sec:.1f}s{star_flag}")
        
        # Save intermediate weights every 5 epochs
        if epoch % 5 == 0 or is_best:
            try:
                ckpt_save_path = Path(args.output_model)
                ckpt_save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_weights, ckpt_save_path)
                # Also save to deployment weights for seamless simulator pickup
                torch.save(best_weights, "/kaggle/working/catcn_deployment_weights.pt")
            except Exception:
                pass
                
    total_train_sec = time.time() - train_start_time
    print(f"\n[MODEL TRAINING COMPLETE] Finished {args.epochs} epochs across all subjects in {total_train_sec/60:.1f} minutes.")
    
    # Load best weights
    model.load_state_dict(best_weights)
    model.eval()
    
    # Save final model
    final_model_path = Path(args.output_model)
    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "montage": args.montage,
        "n_channels": n_ch,
        "hidden_dim": args.hidden_dim,
        "epochs": args.epochs,
        "best_val_loss": best_val_loss,
        "timestamp": datetime.now().isoformat()
    }, final_model_path)
    try:
        torch.save(model.state_dict(), "/kaggle/working/catcn_deployment_weights.pt")
    except Exception:
        pass
    print(f"[CHECKPOINT SAVED] Final Universal Foundation Model saved to: {final_model_path.resolve()}")
    
    # 6. Comprehensive 18-Subject Held-Out Evaluation
    print("\n" + "=" * 96)
    print("  COMPREHENSIVE 18-SUBJECT EVALUATION (ON HELD-OUT PATIENT TRIALS)")
    print("=" * 96)
    print(f"  {'Subject':<14} | {'5.0s Window':<14} | {'10.0s Window':<14} | {'20.0s Window':<14} | {'Polarity Status'}")
    print("  " + "-" * 88)
    
    report_dict = {}
    csv_rows = ["Subject,5s_Standard,5s_PolarityAware,10s_Standard,10s_PolarityAware,20s_Standard,20s_PolarityAware,Status"]
    
    all_5s_std = []
    all_10s_std = []
    all_20s_std = []
    all_10s_pol = []
    
    for sub_name, (te_eeg, te_ya, te_yb) in subject_test_data.items():
        if len(te_eeg) == 0:
            continue
            
        std_5s, pol_5s, n_5s = evaluate_windows(model, te_eeg, te_ya, te_yb, int(5.0 * fs), device)
        std_10s, pol_10s, n_10s = evaluate_windows(model, te_eeg, te_ya, te_yb, int(10.0 * fs), device)
        std_20s, pol_20s, n_20s = evaluate_windows(model, te_eeg, te_ya, te_yb, int(20.0 * fs), device)
        
        all_5s_std.append(std_5s)
        all_10s_std.append(std_10s)
        all_20s_std.append(std_20s)
        all_10s_pol.append(pol_10s)
        
        is_inverted = std_10s < 50.0 and pol_10s >= 55.0
        status_str = "INVERTED DIPOLE" if is_inverted else "NORMAL (Aligned)"
        
        print(f"  {sub_name:<14} | {std_5s:>6.1f}% ({pol_5s:>4.1f}%) | {std_10s:>6.1f}% ({pol_10s:>4.1f}%) | {std_20s:>6.1f}% ({pol_20s:>4.1f}%) | {status_str}")
        
        report_dict[sub_name] = {
            "5s_standard": std_5s, "5s_polarity_aware": pol_5s,
            "10s_standard": std_10s, "10s_polarity_aware": pol_10s,
            "20s_standard": std_20s, "20s_polarity_aware": pol_20s,
            "status": status_str
        }
        csv_rows.append(f"{sub_name},{std_5s:.2f},{pol_5s:.2f},{std_10s:.2f},{pol_10s:.2f},{std_20s:.2f},{pol_20s:.2f},{status_str}")
        
    print("  " + "-" * 88)
    mean_5s = float(np.mean(all_5s_std))
    mean_10s = float(np.mean(all_10s_std))
    mean_20s = float(np.mean(all_20s_std))
    mean_10s_pol = float(np.mean(all_10s_pol))
    
    print(f"  {'GRAND MEAN':<14} | {mean_5s:>6.1f}%         | {mean_10s:>6.1f}% ({mean_10s_pol:>4.1f}%) | {mean_20s:>6.1f}%         | (N={len(all_10s_std)} Subjects)")
    print(f"  {'STD DEV':<14}    | {float(np.std(all_5s_std)):>6.1f}%         | {float(np.std(all_10s_std)):>6.1f}%         | {float(np.std(all_20s_std)):>6.1f}%         |")
    print("=" * 96)
    
    # Save reports
    report_json_path = Path(args.output_report_json)
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_json_path, "w") as f:
        json.dump({
            "grand_mean_5s": mean_5s,
            "grand_mean_10s": mean_10s,
            "grand_mean_20s": mean_20s,
            "grand_mean_10s_polarity_aware": mean_10s_pol,
            "subjects": report_dict
        }, f, indent=2)
        
    report_csv_path = Path(args.output_report_csv)
    with open(report_csv_path, "w") as f:
        f.write("\n".join(csv_rows))
        
    print(f"\n[REPORTS SAVED]:")
    print(f"  -> JSON: {report_json_path.resolve()}")
    print(f"  -> CSV:  {report_csv_path.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal Foundation Model Training for Auditory Attention Decoding")
    parser.add_argument("--montage", type=str, default="near_ear_expanded", help="Electrode montage (default: near_ear_expanded)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs (default: 20)")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size (default: 256)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate (default: 1e-3)")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension (default: 64)")
    parser.add_argument("--window_sec", type=float, default=5.0, help="Training window size in seconds (default: 5.0)")
    parser.add_argument("--hop_sec", type=float, default=2.5, help="Training window hop size in seconds (default: 2.5)")
    parser.add_argument("--test_split", type=float, default=0.20, help="Fraction of trials per subject held out for testing (default: 0.20)")
    parser.add_argument("--output_model", type=str, default="/kaggle/working/catcn_universal_model.pt", help="Path to save universal model checkpoint")
    parser.add_argument("--output_report_json", type=str, default="/kaggle/working/universal_evaluation_report.json", help="Path to save JSON report")
    parser.add_argument("--output_report_csv", type=str, default="/kaggle/working/universal_evaluation_report.csv", help="Path to save CSV report")
    args = parser.parse_args()
    
    run_universal_training(args)
