import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from pathlib import Path
from copy import deepcopy

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.vlaai_lite import VLAAILite
from baselines.ridge_aad import load_subject_examples, subject_files, speech_envelope, iter_leave_one_subject_out

MAPPING = {1: "A", 2: "B"}
FS = 64
COMPRESSION = 0.6
LOWPASS_HZ = 8.0
CHANNELS = [12, 14, 16, 22, 50, 52, 54, 60]  # T7, T8, C3, C4, CP5, CP6, P7, P8

def extract_envelope(wav):
    return speech_envelope(wav, compression=COMPRESSION, lowpass_hz=LOWPASS_HZ, fs=FS, normalize=True)

class PearsonMSELoss(nn.Module):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        # pred, target: [Batch, 1, Time]
        pred_mean = pred.mean(dim=2, keepdim=True)
        target_mean = target.mean(dim=2, keepdim=True)
        pred_std = pred.std(dim=2, keepdim=True) + 1e-8
        target_std = target.std(dim=2, keepdim=True) + 1e-8
        
        cov = ((pred - pred_mean) * (target - target_mean)).mean(dim=2)
        corr = cov / (pred_std.squeeze(2) * target_std.squeeze(2))
        
        pearson_loss = 1 - corr.mean()
        mse_loss = self.mse(pred, target)
        
        return pearson_loss + self.alpha * mse_loss

def prepare_dataset(examples):
    X = []
    Y = []
    Y_A = []
    Y_B = []
    labels = []
    
    for ex in examples:
        eeg = ex.eeg[CHANNELS, :]
        env_a = extract_envelope(ex.wav_a)
        env_b = extract_envelope(ex.wav_b)
        
        min_len = min(eeg.shape[1], len(env_a), len(env_b))
        eeg = eeg[:, :min_len]
        env_a = env_a[:min_len]
        env_b = env_b[:min_len]
        
        target = env_a if MAPPING[ex.label] == "A" else env_b
        
        X.append(eeg)
        Y.append(target)
        Y_A.append(env_a)
        Y_B.append(env_b)
        labels.append(ex.label)
        
    return X, Y, Y_A, Y_B, labels

def evaluate_model(model, X, Y_A, Y_B, labels, device, zero_eeg=False):
    model.eval()
    n_correct = 0
    with torch.no_grad():
        for i in range(len(X)):
            x = torch.FloatTensor(X[i]).unsqueeze(0).to(device)
            if zero_eeg:
                x = torch.zeros_like(x)
            
            pred = model(x).squeeze(0).squeeze(0).cpu().numpy()
            
            env_a = Y_A[i]
            env_b = Y_B[i]
            
            corr_a = np.corrcoef(pred, env_a)[0, 1]
            corr_b = np.corrcoef(pred, env_b)[0, 1]
            
            attended = MAPPING[labels[i]]
            if attended == "A" and corr_a > corr_b:
                n_correct += 1
            elif attended == "B" and corr_b > corr_a:
                n_correct += 1
                
    return n_correct / len(X)

def train_loso():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    paths = subject_files()
    if not paths:
        print("No subjects found.")
        return
        
    subject_examples = {str(p): load_subject_examples(p) for p in paths}
    
    folds = list(iter_leave_one_subject_out(paths))
    all_normal_accs = []
    all_zero_accs = []
    
    for held_out_path, train_paths in folds:
        held_out_key = str(held_out_path)
        print(f"\nEvaluating fold with held-out subject: {held_out_path.stem}")
        
        train_exs = []
        for p in train_paths:
            train_exs.extend(subject_examples[str(p)])
            
        test_exs = subject_examples[held_out_key]
        
        # We need an internal validation set for early stopping.
        # Let's take 10% of train_exs as val_exs
        np.random.shuffle(train_exs)
        val_split = int(0.1 * len(train_exs))
        val_exs = train_exs[:val_split]
        train_exs = train_exs[val_split:]
        
        X_tr, Y_tr, YA_tr, YB_tr, L_tr = prepare_dataset(train_exs)
        X_va, Y_va, YA_va, YB_va, L_va = prepare_dataset(val_exs)
        X_te, Y_te, YA_te, YB_te, L_te = prepare_dataset(test_exs)
        
        # Normalization based on train statistics ONLY
        # Normalize EEG across channels
        eeg_concat = np.concatenate(X_tr, axis=1)
        mean_eeg = eeg_concat.mean(axis=1, keepdims=True)
        std_eeg = eeg_concat.std(axis=1, keepdims=True) + 1e-12
        
        X_tr = [(x - mean_eeg) / std_eeg for x in X_tr]
        X_va = [(x - mean_eeg) / std_eeg for x in X_va]
        X_te = [(x - mean_eeg) / std_eeg for x in X_te]
        
        model = VLAAILite(in_channels=8).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = PearsonMSELoss(alpha=0.1)
        
        best_val_acc = 0.0
        best_weights = deepcopy(model.state_dict())
        patience = 5
        epochs_no_improve = 0
        
        for epoch in range(50):
            model.train()
            train_loss = 0.0
            
            # Simple batching (batch size = 1 for variable lengths)
            for i in range(len(X_tr)):
                x = torch.FloatTensor(X_tr[i]).unsqueeze(0).to(device)
                y = torch.FloatTensor(Y_tr[i]).unsqueeze(0).unsqueeze(0).to(device)
                
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            val_acc = evaluate_model(model, X_va, YA_va, YB_va, L_va, device)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= patience:
                break
                
        model.load_state_dict(best_weights)
        
        # Test Evaluation
        normal_acc = evaluate_model(model, X_te, YA_te, YB_te, L_te, device, zero_eeg=False)
        zero_acc = evaluate_model(model, X_te, YA_te, YB_te, L_te, device, zero_eeg=True)
        
        print(f"  Test Normal EEG Accuracy: {normal_acc:.4f}")
        print(f"  Test Zero EEG Accuracy:   {zero_acc:.4f}")
        
        all_normal_accs.append(normal_acc)
        all_zero_accs.append(zero_acc)
        
    final_normal = np.mean(all_normal_accs)
    final_zero = np.mean(all_zero_accs)
    print("\n" + "="*50)
    print("VLAAI-LITE 8-CHANNEL LOSO RESULTS")
    print("="*50)
    print(f"Overall Normal EEG Accuracy: {final_normal:.4f}")
    print(f"Overall Zero EEG Accuracy:   {final_zero:.4f}")
    
    if final_normal <= final_zero + 0.02:
        print("\n🚨 SANITY CHECK FAILED: Model is not using EEG information.")
    else:
        print("\n✅ SANITY CHECK PASSED: Model is successfully decoding EEG.")

if __name__ == "__main__":
    train_loso()
