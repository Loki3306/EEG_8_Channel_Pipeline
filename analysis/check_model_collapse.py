import sys
import torch
import torch.optim as optim
import numpy as np
from pathlib import Path
from copy import deepcopy

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.vlaai_lite import VLAAILite
from training.train_vlaai_lite_loso import prepare_dataset, PearsonMSELoss
from baselines.ridge_aad import load_subject_examples, subject_files

def run_collapse_diagnostic():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Diagnostic] Using device: {device}")
    
    paths = subject_files()
    if not paths:
        print("No subjects found.")
        return
        
    print("[Diagnostic] Loading subjects...")
    subject_examples = {str(p): load_subject_examples(p) for p in paths}
    
    # We will just train on Subject 1, evaluate on Subject 2
    # to see if the model collapses.
    train_key = str(paths[0])
    test_key = str(paths[1])
    
    train_exs = subject_examples[train_key]
    test_exs = subject_examples[test_key]
    
    X_tr, Y_tr, YA_tr, YB_tr, L_tr = prepare_dataset(train_exs)
    X_te, Y_te, YA_te, YB_te, L_te = prepare_dataset(test_exs)
    
    # Normalize
    eeg_concat = np.concatenate(X_tr, axis=1)
    mean_eeg = eeg_concat.mean(axis=1, keepdims=True)
    std_eeg = eeg_concat.std(axis=1, keepdims=True) + 1e-12
    
    X_tr = [(x - mean_eeg) / std_eeg for x in X_tr]
    X_te = [(x - mean_eeg) / std_eeg for x in X_te]
    
    model = VLAAILite(in_channels=8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = PearsonMSELoss(alpha=0.1)
    
    print("[Diagnostic] Training for 15 epochs on 1 subject...")
    for epoch in range(15):
        model.train()
        for i in range(len(X_tr)):
            x = torch.FloatTensor(X_tr[i]).unsqueeze(0).to(device)
            y = torch.FloatTensor(Y_tr[i]).unsqueeze(0).unsqueeze(0).to(device)
            
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
    print("\n[Diagnostic] Extracting predictions for 10 random test trials...")
    model.eval()
    
    np.random.seed(42)
    indices = np.random.choice(len(X_te), size=10, replace=False)
    
    preds = []
    with torch.no_grad():
        for i in indices:
            x = torch.FloatTensor(X_te[i]).unsqueeze(0).to(device)
            pred = model(x).squeeze().cpu().numpy()
            preds.append(pred)
            
    # Compute cross-correlation matrix
    print("\n[Diagnostic] Cross-Correlation Matrix between 10 trial outputs:")
    
    matrix = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            # Trim to min length
            mlen = min(len(preds[i]), len(preds[j]))
            p1 = preds[i][:mlen]
            p2 = preds[j][:mlen]
            corr = np.corrcoef(p1, p2)[0, 1]
            matrix[i, j] = corr
            
    # Print formatted matrix
    headers = "    " + "".join([f"T{i:<6}" for i in range(10)])
    print(headers)
    for i in range(10):
        row = f"T{i:<2} "
        for j in range(10):
            row += f"{matrix[i, j]:.3f}  "
        print(row)
        
    # Check for collapse
    upper_triangle = matrix[np.triu_indices(10, k=1)]
    mean_cross_corr = np.mean(upper_triangle)
    print(f"\nMean Cross-Correlation between different trials: {mean_cross_corr:.4f}")
    
    if mean_cross_corr > 0.95:
        print("🚨 RESULT: MODEL HAS COLLAPSED. It is outputting nearly identical waveforms regardless of EEG input.")
    elif mean_cross_corr > 0.50:
        print("⚠️ RESULT: PARTIAL COLLAPSE. Outputs are highly correlated with each other.")
    else:
        print("✅ RESULT: MODEL IS REACTIVE. Outputs are distinct across trials.")

if __name__ == "__main__":
    run_collapse_diagnostic()
