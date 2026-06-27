import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.matchnet import ContrastiveMatchNet
from analysis.experiment_16_input_equivalence import get_dtu_tensor
from train_kul_native import get_kul_trials, preprocess_trial, evaluate_val

def find_model(name_substrings):
    for r, d, f in os.walk("/kaggle/input"):
        for file in f:
            if file.endswith(".pth") and all(sub in file.lower() for sub in name_substrings):
                return os.path.join(r, file)
    for r, d, f in os.walk("checkpoints"):
        for file in f:
            if file.endswith(".pth") and all(sub in file.lower() for sub in name_substrings):
                return os.path.join(r, file)
    return None

def evaluate_dtu_fast(model, e_dtu, a_dtu, u_dtu, device):
    model.eval()
    n_correct = 0.0
    n_total = 0
    with torch.no_grad():
        for i in range(len(e_dtu)):
            e = torch.tensor(e_dtu[i], dtype=torch.float32).unsqueeze(0).to(device)
            a = torch.tensor(a_dtu[i], dtype=torch.float32).unsqueeze(0).to(device)
            u = torch.tensor(u_dtu[i], dtype=torch.float32).unsqueeze(0).to(device)
            
            z_eeg, z_a, z_b = model(e, a, u)
            sim_a = torch.nn.functional.cosine_similarity(z_eeg, z_a, dim=1).mean().item()
            sim_b = torch.nn.functional.cosine_similarity(z_eeg, z_b, dim=1).mean().item()
            
            if sim_a > sim_b: n_correct += 1.0
            elif sim_a == sim_b: n_correct += 0.5
            n_total += 1
            
    return n_correct / max(n_total, 1)

def run_cross_eval():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dtu_model_path = find_model(["matchnet", "fold"]) or find_model(["matchnet"])
    kul_model_path = find_model(["matchnet_kul_native"])
    
    if not dtu_model_path:
        print("Missing DTU model. Cannot run cross-evaluation.")
        return
        
    if not kul_model_path:
        print("Missing KUL model. Please run 'train_kul_native.py' first.")
        return
        
    print(f"Loaded DTU Model: {dtu_model_path}")
    print(f"Loaded KUL Model: {kul_model_path}")
    
    model_dtu = ContrastiveMatchNet(eeg_model_type="eegnet", eeg_channels=8, audio_channels=28, latent_dim=64).to(device)
    model_dtu.load_state_dict(torch.load(dtu_model_path, map_location=device))
    
    model_kul = ContrastiveMatchNet(eeg_model_type="eegnet", eeg_channels=8, audio_channels=28, latent_dim=64).to(device)
    model_kul.load_state_dict(torch.load(kul_model_path, map_location=device))
    
    # Load DTU Data
    print("\nLoading DTU Validation Data...")
    e_dtu, a_dtu, u_dtu = get_dtu_tensor()
    if e_dtu is not None:
        # Use a consistent subset if it's too large
        e_dtu, a_dtu, u_dtu = e_dtu[:1000], a_dtu[:1000], u_dtu[:1000]
        
    # Load KUL Val Data
    print("Loading KUL Validation Data...")
    cache_path = "kul_gammatone_cache.pkl"
    if os.path.exists(cache_path):
        import pickle
        with open(cache_path, "rb") as f:
            envelope_cache = pickle.load(f)
    else:
        print("Missing kul_gammatone_cache.pkl")
        return
        
    trials = get_kul_trials()
    val_trials = trials[15:]
    val_data = []
    for t in val_trials:
        x, ya, yb = preprocess_trial(t, envelope_cache, apply_car=True)
        if x is not None:
            val_data.append((x, ya, yb))
            
    print("\nEvaluating DTU -> DTU...")
    acc_dd = evaluate_dtu_fast(model_dtu, e_dtu, a_dtu, u_dtu, device)
    
    print("Evaluating DTU -> KUL...")
    acc_dk = evaluate_val(model_dtu, val_data, device, window_sec=10)
    
    print("Evaluating KUL -> KUL...")
    acc_kk = evaluate_val(model_kul, val_data, device, window_sec=10)
    
    print("Evaluating KUL -> DTU...")
    acc_kd = evaluate_dtu_fast(model_kul, e_dtu, a_dtu, u_dtu, device)
    
    print("\n================================================================================")
    print("PHASE I: 2x2 CROSS-DATASET EVALUATION MATRIX")
    print("================================================================================")
    print(f"| {'Train Set':<10} | {'Test Set':<10} | {'Accuracy':<10} |")
    print("-" * 40)
    print(f"| {'DTU':<10} | {'DTU':<10} | {acc_dd*100:>9.2f}% | (Reference Ceiling)")
    print(f"| {'DTU':<10} | {'KUL':<10} | {acc_dk*100:>9.2f}% | (Zero-Shot Transfer)")
    print(f"| {'KUL':<10} | {'KUL':<10} | {acc_kk*100:>9.2f}% | (Native Ceiling)")
    print(f"| {'KUL':<10} | {'DTU':<10} | {acc_kd*100:>9.2f}% | (Reverse Transfer)")
    print("================================================================================")
    
if __name__ == "__main__":
    run_cross_eval()
