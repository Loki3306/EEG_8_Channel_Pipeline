import os
import sys
import torch
import torch.nn.functional as F
import numpy as np

# Adjust path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.aad_conformer import AADConformer
from data.kul_cached_dataset import KULCachedLoader
from src.confidence.selective_predictor import SelectivePredictor
from src.confidence.selective_metrics import calculate_selective_risk
from torch.utils.data import DataLoader, TensorDataset

def run_selective_pipeline(subject="S1", model_path=None, threshold=0.70):
    print("=" * 80)
    print(f"PHASE 11 — SELECTIVE AAD PIPELINE (Subject {subject})")
    print("=" * 80)
    print(f"Configuration:")
    print(f" - Threshold: {threshold}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" - Device: {device}")
    
    # Load data
    print(f"Loading KUL Data for {subject}...")
    try:
        data_path = "/kaggle/input/datasets/lokeshgile/kul-processed/data/processed_kul"
        if not os.path.exists(data_path):
            data_path = "data/processed_kul" # Fallback local
            
        loader = KULCachedLoader(data_path)
        all_data = loader.load_all()
        if subject not in all_data:
            print(f"Subject {subject} not found in data.")
            return
            
        subject_trials = all_data[subject]
        
        # Structure the trials like before
        eeg, wav_a, wav_b, labels = [], [], [], []
        for trial in subject_trials:
            eeg.append(trial["eeg"])
            wav_a.append(trial["audio_a"])
            wav_b.append(trial["audio_b"])
            # KUL cache always sets audio_a to the attended audio, so label is 0
            labels.append(0)
            
        print(f"Data loaded: Trials {len(labels)}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Failed to load data: {e}")
        return
        
    # Load model
    print("Initializing AADConformer...")
    model = AADConformer(eeg_channels=8, audio_channels=28, num_classes=2).to(device)
    model.eval()
    
    if model_path and os.path.exists(model_path):
        print(f"Loading weights from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"WARNING: Model path not found. Using untrained weights for demonstration.")
        
    # Initialize predictor
    predictor = SelectivePredictor(threshold=threshold)
    
    # Metrics tracking
    trial_results = []
    all_window_results = []
    
    print("\nStarting inference...")
    
    # Process trial by trial
    with torch.no_grad():
        for t_idx in range(len(eeg)):
            t_eeg = eeg[t_idx]
            t_wa = wav_a[t_idx]
            t_wb = wav_b[t_idx]
            label = labels[t_idx]
            
            # Simple windowing (5s = 320 samples at 64Hz)
            win_samples = 320
            
            window_results = []
            
            for start in range(0, t_eeg.shape[1] - win_samples + 1, win_samples):
                end = start + win_samples
                
                # Extract chunk
                c_eeg = torch.FloatTensor(t_eeg[:, start:end]).unsqueeze(0).to(device)
                c_wa = torch.FloatTensor(t_wa[:, start:end]).unsqueeze(0).to(device)
                c_wb = torch.FloatTensor(t_wb[:, start:end]).unsqueeze(0).to(device)
                
                # Forward
                pred_idx, margin, _ = model.predict_confidence(c_eeg, c_wa, c_wb)
                
                # For this demo we don't extract pearson locally, we just use margin confidence
                res = predictor.predict_window(margin)
                res["ground_truth"] = label
                window_results.append(res)
                all_window_results.append(res)
                
            # Trial aggregation
            trial_res = predictor.predict_trial(window_results, aggregation="majority")
            trial_res["ground_truth"] = label
            trial_res["trial_idx"] = t_idx
            trial_results.append(trial_res)
            
            if t_idx < 3: # Print first 3 trials explicitly as requested by "Verbose"
                print(f"  Trial {t_idx} -> Accepted Windows: {trial_res['accepted_windows_count']}/{trial_res['total_windows_count']}, Trial Accepted: {trial_res['accepted']}, Pred: {trial_res['prediction']}, Truth: {label}")
                
    # Calculate aggregate metrics
    print("\n" + "=" * 80)
    print("SELECTIVE METRICS REPORT")
    print("=" * 80)
    
    # Trial Level
    t_truths = [t["ground_truth"] for t in trial_results]
    t_preds = [t["prediction"] for t in trial_results]
    t_confs = [t["confidence"] for t in trial_results]
    
    metrics = calculate_selective_risk(t_truths, t_preds, t_confs, threshold)
    print(f"Trial-Level Results (Threshold = {threshold}):")
    print(f"  Coverage:          {metrics['coverage']*100:.1f}%")
    print(f"  Accepted Accuracy: {metrics['accepted_accuracy']*100:.1f}%")
    print(f"  Rejected Accuracy: {metrics['rejected_accuracy']*100:.1f}%")
    print(f"  Selective Risk:    {metrics['selective_risk']:.4f}")
    print(f"  Accepted Trials:   {metrics['accepted_count']}")
    print(f"  Rejected Trials:   {metrics['rejected_count']}")
    
    print("\nPipeline execution complete.")
    
if __name__ == "__main__":
    # Test path: /kaggle/working/EEG_8_Channel_Pipeline/results/run7_multitask_conformer_loso/checkpoints/seed_1/model_S1.pt
    model_path = "/kaggle/working/EEG_8_Channel_Pipeline/results/run7_multitask_conformer_loso/checkpoints/seed_1/model_S1.pt"
    run_selective_pipeline(subject="S1", model_path=model_path, threshold=0.70)
