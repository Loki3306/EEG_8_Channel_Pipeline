import os
import sys
import numpy as np
import torch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.tcnn import TCNN
from data.kul_cached_dataset import KULCachedLoader
import torch.nn.functional as F

FS = 64
DECISION_WINDOW_SEC = 10

def analyze_subject(subject_id, model, test_data, device):
    model.eval()
    win_samples = int(DECISION_WINDOW_SEC * FS)
    
    print(f"\n==================================================")
    print(f"Subject: {subject_id}")
    print(f"==================================================")
    print(f"{'Trial':<8} | {'GT':<10} | {'Majority':<10} | {'Win Votes (T1:T2)':<20} | {'Result'}")
    print("-" * 75)
    
    pred_t1 = 0
    pred_t2 = 0
    
    conf_matrix = {
        (0, 0): 0, # GT 1 -> Pred 1
        (0, 1): 0, # GT 1 -> Pred 2
        (1, 0): 0, # GT 2 -> Pred 1
        (1, 1): 0  # GT 2 -> Pred 2
    }
    
    with torch.no_grad():
        for i, t in enumerate(test_data):
            x = t["eeg"].numpy()
            meta = t["meta"]
            attended_track = str(meta.get('attended_track', 'Unknown'))
            if attended_track not in ['1', '2']:
                continue
                
            true_label = 0 if attended_track == '1' else 1
            
            start = 0
            trial_logits = []
            
            win_pred_1 = 0
            win_pred_2 = 0
            win_string = ""
            
            while start + win_samples <= x.shape[1]:
                end = start + win_samples
                cx = torch.FloatTensor(x[:, start:end]).unsqueeze(0).to(device)
                
                logits = model(cx)
                probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                pred_label = np.argmax(probs)
                
                if pred_label == 0:
                    win_pred_1 += 1
                    win_string += "1"
                else:
                    win_pred_2 += 1
                    win_string += "2"
                    
                trial_logits.append(logits.squeeze(0).cpu().numpy())
                start += win_samples
                
            if trial_logits:
                mean_logits = np.mean(trial_logits, axis=0)
                trial_pred = int(np.argmax(mean_logits))
                
                gt_str = f"Track {true_label + 1}"
                pred_str = f"Track {trial_pred + 1}"
                result_str = "CORRECT" if trial_pred == true_label else "WRONG"
                
                print(f"{i+1:02d}       | {gt_str:<10} | {pred_str:<10} | {win_string:<20} | {result_str}")
                
                if trial_pred == 0:
                    pred_t1 += 1
                else:
                    pred_t2 += 1
                    
                conf_matrix[(true_label, trial_pred)] += 1
                
    print("\n--- Confusion Matrix ---")
    print(f"GT Track1 -> Pred Track1: {conf_matrix[(0, 0)]}")
    print(f"GT Track1 -> Pred Track2: {conf_matrix[(0, 1)]}")
    print(f"GT Track2 -> Pred Track1: {conf_matrix[(1, 0)]}")
    print(f"GT Track2 -> Pred Track2: {conf_matrix[(1, 1)]}")
    
    print("\n--- Total Predictions ---")
    print(f"Predicted Track1 = {pred_t1}")
    print(f"Predicted Track2 = {pred_t2}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading KUL Cache...")
    loader = KULCachedLoader(REPO_ROOT / "data" / "processed_kul")
    all_subject_data = loader.load_all()
    
    subjects_to_test = ["S1", "S10", "S12", "S13"]
    
    model_dir = REPO_ROOT / "results" / "tcnn_loso"
    
    for sub in subjects_to_test:
        model_path = model_dir / f"best_model_{sub}.pt"
        if not model_path.exists():
            print(f"Model checkpoint for {sub} not found at {model_path}")
            continue
            
        model = TCNN(in_channels=8, num_classes=2).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        analyze_subject(sub, model, all_subject_data[sub], device)

if __name__ == "__main__":
    main()
