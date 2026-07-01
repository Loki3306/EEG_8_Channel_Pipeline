import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Adjust path if needed
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.aad_conformer import AADConformer
from data.kul_cached_dataset import KULCachedLoader
from src.confidence.selective_predictor import SelectivePredictor
from src.confidence.selective_metrics import calculate_selective_risk
from analysis.run7_multitask_evaluation import extract_subject_predictions

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
            data_path = REPO_ROOT / "data" / "processed_kul" # Fallback local
            
        loader = KULCachedLoader(data_path)
        all_data = loader.load_all()
        if subject not in all_data:
            print(f"Subject {subject} not found in data.")
            return
            
        test_trials = all_data[subject]
        print(f"Data loaded: Trials {len(test_trials)}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Failed to load data: {e}")
        return
        
    # Load model
    print("Initializing AADConformer...")
    model = AADConformer(in_channels=8).to(device)
    model.eval()
    
    if model_path and os.path.exists(model_path):
        print(f"Loading weights from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"WARNING: Model path not found. Using untrained weights for demonstration.")
        
    # Initialize predictor
    predictor = SelectivePredictor(threshold=threshold)
    
    print("\nStarting inference via canonical extract_subject_predictions (10s windows, normalized)...")
    
    fs = 64
    win_samples = 10 * fs
    hop_samples = fs
    
    # The canonical function evaluates all windows for all trials and returns flat lists.
    # But for Selective Trial Aggregation, we need to group windows by trial.
    # Let's extract predictions trial by trial using the canonical function.
    
    trial_results = []
    all_window_results = []
    
    with torch.no_grad():
        for t_idx, trial in enumerate(test_trials):
            # Pass a single trial in a list to extract_subject_predictions
            m_cln, c_cln, conf_cln = extract_subject_predictions(
                model, [trial], device, win_samples, hop_samples, mode="clean"
            )
            
            # KUL cache always sets audio_a to the attended audio, so label is 1 (True = Audio A)
            # Actually, extract_subject_predictions returns `correct` boolean list
            # which is True if margin > 0. Since Audio A is always correct in KUL dataset, 
            # margin > 0 means it correctly chose A.
            
            window_results = []
            
            for w_idx in range(len(m_cln)):
                margin = m_cln[w_idx]
                c_prob = conf_cln[w_idx]
                is_correct = c_cln[w_idx]
                
                # In the old predictor, predict_window expected pearson_a, pearson_b, etc.
                # Here we just have margin and c_prob. The SelectivePredictor just uses them.
                res = predictor.predict_window(
                    margin, 
                    pearson_a=margin, # Dummy values if not used in thresholding
                    pearson_b=0.0,
                    use_pearson=True,
                    learned_confidence=c_prob
                )
                
                # Overwrite prediction to match correct boolean
                # Since A is ground truth (1), if correct then prediction is 1, else 0
                pred = 1 if is_correct else 0
                res["prediction"] = pred
                res["ground_truth"] = 1
                
                window_results.append(res)
                all_window_results.append(res)
                
            # Trial aggregation
            # Trial decision logic:
            trial_res = predictor.predict_trial(window_results, aggregation="majority", min_accept_ratio=0.50)
            trial_res["ground_truth"] = 1
            trial_res["trial_idx"] = t_idx
            trial_results.append(trial_res)
            
            if t_idx == 0:
                print("\n" + "-" * 40)
                print(f"TRIAL 0 TRACE (Threshold {threshold})")
                print("-" * 40)
                for w_idx, w in enumerate(window_results):
                    print(f"Window {w_idx+1:<2}: Confidence={w['confidence']:.4f}, Margin={w['margin']:.4f}, Pred={w['prediction']}, Truth={1}, Accepted={w['accepted']}, Correct={w['prediction']==1}")
                
                print("-" * 40)
                print(f"Threshold:                {threshold}")
                print(f"Accepted windows:         {trial_res['accepted_windows_count']}/{trial_res['total_windows_count']}")
                print(f"Mean window confidence:   {trial_res.get('mean_window_confidence', 0.0):.4f}")
                print(f"Median window confidence: {trial_res.get('median_window_confidence', 0.0):.4f}")
                print(f"Trial decision reason:    {trial_res.get('reason', 'N/A')}")
                print(f"Trial prediction:         {trial_res['prediction']}")
                print(f"Trial correctness:        {trial_res['prediction'] == 1}")
                print("-" * 40 + "\n")
                
    # Function to print report
    def print_report(t_results, title, thresh):
        t_truths = [t["ground_truth"] for t in t_results]
        t_preds = [t["prediction"] for t in t_results]
        t_confs = [t["confidence"] for t in t_results]
        
        metrics = calculate_selective_risk(t_truths, t_preds, t_confs, thresh)
        print(f"\n{title} (Threshold = {thresh}):")
        print(f"  Trial Coverage:          {metrics['coverage']*100:.1f}%")
        print(f"  Overall Accuracy:        {metrics['overall_accuracy']*100:.1f}%")
        print(f"  Accepted Accuracy:       {metrics['accepted_accuracy']*100:.1f}%")
        print(f"  Rejected Accuracy:       {metrics['rejected_accuracy']*100:.1f}%")
        print(f"  Selective Risk:          {metrics['selective_risk']:.4f}  <-- (Defined as 1.0 - Accepted Accuracy)")
        print(f"  Accepted Trials:         {metrics['accepted_count']}/{metrics['total_count']}")
        print(f"  Rejected Trials:         {metrics['rejected_count']}")
                
    print("\n" + "=" * 80)
    print("SELECTIVE METRICS REPORT")
    print("=" * 80)
    
    print_report(trial_results, "Baseline Results", 0.00)
    print_report(trial_results, "Selective Results", threshold)
    
    print("\nPipeline execution complete.")
    
if __name__ == "__main__":
    # Test path: /kaggle/working/EEG_8_Channel_Pipeline/results/run7_multitask_conformer_loso/checkpoints/seed_1/model_S1.pt
    model_path = "/kaggle/working/EEG_8_Channel_Pipeline/results/run7_multitask_conformer_loso/checkpoints/seed_1/model_S1.pt"
    run_selective_pipeline(subject="S1", model_path=model_path, threshold=0.55)
