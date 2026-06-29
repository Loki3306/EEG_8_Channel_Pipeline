import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.kul_cached_dataset import KULCachedLoader
from models.contrastive_aad import ContrastiveAADModel

class ContrastiveDataset(Dataset):
    def __init__(self, subject_data_dict, test_sub, window_sec=10.0, fs=64, steps_per_epoch=25, batch_size=64):
        self.trials = []
        for sub, trials in subject_data_dict.items():
            if sub not in test_sub:
                self.trials.extend(trials)
                
        self.win_samples = int(window_sec * fs)
        self.num_samples = steps_per_epoch * batch_size
        
        self.std_trials = []
        for t in self.trials:
            eeg = t["eeg"]
            a = t["audio_a"]
            b = t["audio_b"]
            
            eeg = (eeg - eeg.mean(dim=1, keepdim=True)) / (eeg.std(dim=1, keepdim=True) + 1e-12)
            a = (a - a.mean(dim=1, keepdim=True)) / (a.std(dim=1, keepdim=True) + 1e-12)
            b = (b - b.mean(dim=1, keepdim=True)) / (b.std(dim=1, keepdim=True) + 1e-12)
            
            self.std_trials.append((eeg, a, b))

    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        trial_idx = torch.randint(0, len(self.std_trials), (1,)).item()
        eeg, a, b = self.std_trials[trial_idx]
        
        max_start = eeg.shape[1] - self.win_samples
        start = torch.randint(0, max_start + 1, (1,)).item()
        end = start + self.win_samples
        
        return eeg[:, start:end], a[:, start:end], b[:, start:end]

def extract_evaluation_windows(all_subject_data, test_subs, window_sec=10.0, hop_sec=10.0, fs=64):
    win_samples = int(window_sec * fs)
    hop_samples = int(hop_sec * fs)
    windows = []
    
    for sub in test_subs:
        if sub not in all_subject_data:
            continue
            
        for t in all_subject_data[sub]:
            eeg = t["eeg"]
            a = t["audio_a"]
            b = t["audio_b"]
            meta = t.get("meta", {})
            
            eeg = (eeg - eeg.mean(dim=1, keepdim=True)) / (eeg.std(dim=1, keepdim=True) + 1e-12)
            a = (a - a.mean(dim=1, keepdim=True)) / (a.std(dim=1, keepdim=True) + 1e-12)
            b = (b - b.mean(dim=1, keepdim=True)) / (b.std(dim=1, keepdim=True) + 1e-12)
            
            att_ear = meta.get("attended_ear", "Unknown")
            att_story = meta.get("stimuli_left", "Unknown") if att_ear == "L" else meta.get("stimuli_right", "Unknown")
            
            start = 0
            while start + win_samples <= eeg.shape[1]:
                end = start + win_samples
                windows.append({
                    "subject": sub,
                    "attended_ear": att_ear,
                    "attended_story": att_story,
                    "eeg": eeg[:, start:end],
                    "audio_a": a[:, start:end],
                    "audio_b": b[:, start:end]
                })
                start += hop_samples
                
    return windows

def get_embeddings(model, windows, device, batch_size=128):
    model.eval()
    all_e, all_a, all_b = [], [], []
    
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch = windows[i:i+batch_size]
            
            eeg = torch.stack([w["eeg"] for w in batch]).to(device)
            aud_a = torch.stack([w["audio_a"] for w in batch]).to(device)
            aud_b = torch.stack([w["audio_b"] for w in batch]).to(device)
            
            e_rep, a_rep = model.get_representations(eeg, aud_a)
            _, b_rep = model.get_representations(eeg, aud_b)
            
            e_rep = F.normalize(e_rep, p=2, dim=1)
            a_rep = F.normalize(a_rep, p=2, dim=1)
            b_rep = F.normalize(b_rep, p=2, dim=1)
            
            all_e.append(e_rep.cpu().numpy())
            all_a.append(a_rep.cpu().numpy())
            all_b.append(b_rep.cpu().numpy())
            
    return np.concatenate(all_e), np.concatenate(all_a), np.concatenate(all_b)

def get_collapse_metrics(E):
    """
    Computes effective rank, total variance (trace of cov), and mean std per dimension.
    """
    cov = np.cov(E.T)
    trace = np.trace(cov)
    dim_std = np.std(E, axis=0).mean()
    
    eigvals = np.linalg.eigvals(cov).real
    eigvals = np.maximum(eigvals, 1e-12)
    p = eigvals / np.sum(eigvals)
    er = np.exp(-np.sum(p * np.log(p)))
    
    return trace, dim_std, er

def run_diagnostics(E, A, B, metadata):
    """
    Computes the 4 high-ROI diagnostics specified by the user.
    """
    diagnostics = {}
    
    # 1. Margin
    cos_att = (E * A).sum(axis=-1)
    cos_unatt = (E * B).sum(axis=-1)
    margin = cos_att - cos_unatt
    diagnostics["Margin_Mean"] = margin.mean()
    diagnostics["Margin_Std"] = margin.std()
    
    # Collapse Metrics
    trace, dim_std, er = get_collapse_metrics(E)
    diagnostics["Cov_Trace"] = trace
    diagnostics["Mean_Dim_Std"] = dim_std
    diagnostics["Effective_Rank"] = er
    
    # SVM Probes on EEG ONLY
    labels_ear = (metadata["ear"] == "L").astype(int)
    labels_subj = metadata["subject"].astype('category').cat.codes
    labels_story = metadata["story"].astype('category').cat.codes
    
    # Use cross-validation to get unbiased estimate
    clf = LinearSVC(max_iter=2000, dual=False)
    
    diagnostics["Acc_Attention"] = cross_val_score(clf, E, labels_ear, cv=5).mean()
    
    # Only run story/subject probes if there are more than 1 class
    if len(np.unique(labels_story)) > 1:
        diagnostics["Acc_Story"] = cross_val_score(clf, E, labels_story, cv=5).mean()
    else:
        diagnostics["Acc_Story"] = np.nan
        
    if len(np.unique(labels_subj)) > 1:
        diagnostics["Acc_Subject"] = cross_val_score(clf, E, labels_subj, cv=5).mean()
    else:
        diagnostics["Acc_Subject"] = np.nan
        
    return diagnostics

def main():
    print("="*70)
    print("   HIGH-ROI EMBEDDING DIAGNOSTICS SUITE")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Fixed smoke parameters
    batch_size = 64
    epochs = 5
    lr = 3e-4
    steps_per_epoch = 25
    test_subs = ["S1", "S9"]

    out_dir = REPO_ROOT / "results" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load Data
    loader = KULCachedLoader(REPO_ROOT / "data" / "processed_kul")
    try:
        all_subject_data = loader.load_all()
    except FileNotFoundError:
        print("Data cache not found.")
        return

    # Extract Evaluation Windows
    print("\nExtracting Evaluation Windows...")
    eval_windows = extract_evaluation_windows(all_subject_data, test_subs)
    metadata = pd.DataFrame([{
        "subject": w["subject"],
        "story": w["attended_story"],
        "ear": w["attended_ear"]
    } for w in eval_windows])
    print(f"Total Evaluation Windows: {len(eval_windows)}")

    model = ContrastiveAADModel().to(device)
    
    all_results = []
    
    # -------------------------------------------------------------
    # EPOCH 0
    # -------------------------------------------------------------
    print("\n--- Running Epoch 0 Diagnostics ---")
    E0, A0, B0 = get_embeddings(model, eval_windows, device)
    diag0 = run_diagnostics(E0, A0, B0, metadata)
    diag0["Epoch"] = 0
    all_results.append(diag0)
    
    # -------------------------------------------------------------
    # PRETRAINING (5 Epochs)
    # -------------------------------------------------------------
    print("\nPretraining for 5 Epochs...")
    train_ds = ContrastiveDataset(all_subject_data, test_subs, steps_per_epoch=steps_per_epoch, batch_size=batch_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for eeg_batch, a_pos_batch, a_neg_batch in train_loader:
            eeg_batch = eeg_batch.to(device)
            a_pos_batch = a_pos_batch.to(device)
            a_neg_batch = a_neg_batch.to(device)
            
            optimizer.zero_grad()
            loss = model(eeg_batch, a_pos_batch, a_neg_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch:2d}/{epochs} | InfoNCE Loss: {total_loss/len(train_loader):.4f}")

    # -------------------------------------------------------------
    # EPOCH 5
    # -------------------------------------------------------------
    print("\n--- Running Epoch 5 Diagnostics ---")
    E5, A5, B5 = get_embeddings(model, eval_windows, device)
    diag5 = run_diagnostics(E5, A5, B5, metadata)
    diag5["Epoch"] = 5
    all_results.append(diag5)
    
    # -------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------
    df = pd.DataFrame(all_results)
    
    print("\n" + "="*70)
    print("   DIAGNOSTIC RESULTS SUMMARY")
    print("="*70)
    
    for e in [0, 5]:
        row = df[df["Epoch"] == e].iloc[0]
        print(f"\n[Epoch {e}]")
        print(f"  Margin Separation:   {row['Margin_Mean']:.4f} (std: {row['Margin_Std']:.4f})")
        print(f"  EEG->Attention Acc:  {row['Acc_Attention']*100:.1f}%")
        print(f"  EEG->Story Acc:      {row['Acc_Story']*100:.1f}%")
        print(f"  EEG->Subject Acc:    {row['Acc_Subject']*100:.1f}%")
        print(f"  Effective Rank:      {row['Effective_Rank']:.1f}")
        print(f"  Mean Dim Std:        {row['Mean_Dim_Std']:.4f}")
        
    csv_path = out_dir / "high_roi_diagnostics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved full metrics to {csv_path}")

if __name__ == "__main__":
    main()
