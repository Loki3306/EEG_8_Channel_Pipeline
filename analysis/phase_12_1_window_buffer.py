import os
import sys
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Adjust path if needed
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.aad_conformer import AADConformer
from data.kul_cached_dataset import KULCachedLoader
from decision_engine.window_buffer import WindowPrediction, SequentialWindowBuffer
from analysis.interpretability.utils import safe_corr_np, normalize_eeg, normalize_audio
import argparse

def custom_sliding_window_evaluation(model, eeg, wav_a, wav_b, win_samples, hop_samples):
    """
    Evaluates a single trial using sliding windows, returning full metrics for the sequential buffer.
    """
    windows = []
    
    # Pre-normalize for canonical evaluation
    eeg = normalize_eeg(eeg)
    wav_a = normalize_audio(wav_a)
    wav_b = normalize_audio(wav_b)
    
    min_len = min(eeg.shape[-1], wav_a.shape[-1], wav_b.shape[-1])
    eeg, wav_a, wav_b = eeg[:,:,:min_len], wav_a[:,:,:min_len], wav_b[:,:,:min_len]
    
    window_index = 0
    for start in range(0, eeg.shape[-1] - win_samples + 1, hop_samples):
        stop = start + win_samples
        eeg_win = eeg[:, :, start:stop]
        
        pred, z_pool = model(eeg_win, return_features=True)
        pred = pred.squeeze(0).cpu().numpy()
        
        wa = wav_a[:, :, start:stop].squeeze(1).squeeze(0).cpu().numpy()
        wb = wav_b[:, :, start:stop].squeeze(1).squeeze(0).cpu().numpy()
        
        ca = safe_corr_np(pred, wa)
        cb = safe_corr_np(pred, wb)
        margin = ca - cb
        
        ca_t = torch.tensor([ca], dtype=torch.float32, device=eeg.device)
        cb_t = torch.tensor([cb], dtype=torch.float32, device=eeg.device)
        margin_t = torch.tensor([margin], dtype=torch.float32, device=eeg.device)
        
        conf = model.predict_confidence(z_pool, ca_t, cb_t, margin_t)
        conf = conf.squeeze().item()
        
        prediction = 1 if margin > 0 else 0
        correct = (margin > 0)
        
        windows.append({
            "window_index": window_index,
            "prediction": prediction,
            "confidence": conf,
            "margin": margin,
            "corr_a": ca,
            "corr_b": cb,
            "correct": correct
        })
        window_index += 1
        
    return windows

def validate_infrastructure():
    # Infrastructure Validation
    print("\n[VALIDATION]")
    test_buffer = SequentialWindowBuffer()
    wp1 = WindowPrediction(0, 0.0, "t1", 1, 0.8, 0.1, 0.5, 0.4, True, True)
    wp2 = WindowPrediction(1, 0.1, "t1", 0, 0.4, -0.2, 0.2, 0.4, True, False)
    
    test_buffer.append(wp1)
    test_buffer.append(wp2)
    
    assert test_buffer.length() == 2, "Append failed"
    assert test_buffer.get_last(1)[0].window_index == 1, "last_n() failed"
    assert test_buffer.get_last(5)[0].window_index == 0, "last_n() out of bounds failed"
    assert abs(test_buffer.running_mean_confidence() - 0.6) < 1e-6, "Running mean conf failed"
    assert abs(test_buffer.running_mean_margin() - (-0.05)) < 1e-6, "Running mean margin failed"
    assert test_buffer.running_accuracy() == 0.5, "Running accuracy failed"
    
    test_buffer.reset()
    assert test_buffer.length() == 0, "Reset failed"
    
    print("All infrastructure validations passed.")

def run_phase_12_1_validation(ckpt_path_arg=None):
    print("================================================================================")
    print("PHASE 12.1 — SEQUENTIAL WINDOW BUFFER VALIDATION")
    print("================================================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    out_dir = REPO_ROOT / "results" / "phase12_window_buffer"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    data_path = "/kaggle/input/datasets/lokeshgile/kul-processed/data/processed_kul"
    if not os.path.exists(data_path):
        data_path = REPO_ROOT / "data" / "processed_kul"
        
    try:
        loader = KULCachedLoader(data_path)
        all_data = loader.load_all()
    except Exception as e:
        print(f"Failed to load data: {e}")
        validate_infrastructure()
        return
        
    subject = "S1"
    if subject not in all_data:
        print(f"Subject {subject} not found.")
        validate_infrastructure()
        return
        
    test_trials = all_data[subject]
    print(f"Loaded {len(test_trials)} trials for {subject}.")
    
    model = AADConformer(in_channels=8).to(device)
    model.eval()
    
    if ckpt_path_arg and Path(ckpt_path_arg).exists():
        ckpt_path = Path(ckpt_path_arg)
    else:
        # 1. Local path
        ckpt_path = REPO_ROOT / "results" / "run7_multitask_conformer_loso" / "checkpoints" / "seed_1" / "model_S1.pt"
        
        # 2. Hardcoded Kaggle working path (if trained in the same session)
        if not ckpt_path.exists():
            ckpt_path = Path("/kaggle/working/EEG_8_Channel_Pipeline/results/run7_multitask_conformer_loso/checkpoints/seed_1/model_S1.pt")
            
        # 3. Hardcoded Kaggle input path (if uploaded as a dataset)
        if not ckpt_path.exists():
            # Sometimes Kaggle datasets are mounted under /kaggle/input/eeg-8-channel-pipeline or similar
            possible = list(Path("/kaggle/input").rglob("model_S1.pt"))
            if possible:
                ckpt_path = possible[0]

    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded frozen Conformer checkpoint from {ckpt_path}")
    else:
        print("WARNING: Checkpoint not found! It looked for:")
        print(" - /kaggle/working/EEG_8_Channel_Pipeline/results/run7_multitask_conformer_loso/checkpoints/seed_1/model_S1.pt")
        print(" - Any 'model_S1.pt' inside /kaggle/input/")
        print("Using untrained weights.")
        
    buffer = SequentialWindowBuffer()
    fs = 64
    win_samples = 10 * fs
    hop_samples = fs
    
    trace_data = []
    
    with torch.no_grad():
        for t_idx, trial in enumerate(test_trials):
            eeg = trial["eeg"].unsqueeze(0).to(device)
            wav_a = trial["audio_a"].unsqueeze(0).to(device).mean(dim=1, keepdim=True)
            wav_b = trial["audio_b"].unsqueeze(0).to(device).mean(dim=1, keepdim=True)
            
            windows = custom_sliding_window_evaluation(model, eeg, wav_a, wav_b, win_samples, hop_samples)
            
            # Reset buffer for new trial to isolate temporal memory per trial
            buffer.reset()
            
            for w in windows:
                current_time = time.time()
                
                wp = WindowPrediction(
                    window_index=w["window_index"],
                    timestamp=current_time,
                    trial_id=str(t_idx),
                    prediction=w["prediction"],
                    confidence=w["confidence"],
                    margin=w["margin"],
                    corr_a=w["corr_a"],
                    corr_b=w["corr_b"],
                    accepted=True,  # No decision logic yet
                    correct=w["correct"]
                )
                
                buffer.append(wp)
                
                trace_data.append({
                    "subject": subject,
                    "trial": t_idx,
                    "window": wp.window_index,
                    "prediction": wp.prediction,
                    "confidence": wp.confidence,
                    "margin": wp.margin,
                    "corr_a": wp.corr_a,
                    "corr_b": wp.corr_b,
                    "accepted": wp.accepted,
                    "correct": wp.correct,
                    "timestamp": wp.timestamp
                })
                
            print("-" * 33)
            print(f"Buffer Summary [Trial {t_idx}]")
            confs = buffer.confidence_history()
            print(f"Mean confidence   : {np.mean(confs):.4f}")
            print(f"Median confidence : {np.median(confs):.4f}")
            print(f"Std confidence    : {np.std(confs):.4f}")
            print(f"Mean margin       : {buffer.running_mean_margin():.4f}")
            
            preds = buffer.prediction_history()
            count_1 = preds.count(1)
            count_0 = preds.count(0)
            print(f"Prediction counts : {{1: {count_1}, 0: {count_0}}}")
            print(f"Acceptance counts : {buffer.length()}")
            print(f"Window count      : {buffer.length()}")
            print("\n")
            
    df = pd.DataFrame(trace_data)
    csv_path = out_dir / "buffer_trace.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved sequential trace to {csv_path}")
    validate_infrastructure()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to the model checkpoint")
    args = parser.parse_args()
    run_phase_12_1_validation(args.ckpt_path)
