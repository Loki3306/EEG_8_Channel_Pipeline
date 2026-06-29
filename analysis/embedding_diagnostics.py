import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

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
    """
    Extracts non-overlapping windows from specific test subjects and maintains metadata.
    """
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
            
            # L2 Normalize for cosine similarity
            e_rep = F.normalize(e_rep, p=2, dim=1)
            a_rep = F.normalize(a_rep, p=2, dim=1)
            b_rep = F.normalize(b_rep, p=2, dim=1)
            
            all_e.append(e_rep.cpu().numpy())
            all_a.append(a_rep.cpu().numpy())
            all_b.append(b_rep.cpu().numpy())
            
    return np.concatenate(all_e), np.concatenate(all_a), np.concatenate(all_b)

def compute_retrieval_metrics(E, A):
    """
    Given E [N, D] and A [N, D], each E[i] corresponds to A[i].
    Returns retrieval metrics.
    """
    N = E.shape[0]
    # Similarity matrix: sim[i, j] = cosine(E[i], A[j])
    sim = E @ A.T
    
    ranks = []
    for i in range(N):
        # Sort indices by descending similarity
        sorted_idx = np.argsort(-sim[i])
        # Find rank of the correct audio
        rank = np.where(sorted_idx == i)[0][0] + 1
        ranks.append(rank)
        
    ranks = np.array(ranks)
    r1 = (ranks == 1).mean()
    r5 = (ranks <= 5).mean()
    median_r = np.median(ranks)
    mean_r = np.mean(ranks)
    return r1, r5, median_r, mean_r

def effective_rank(covariance_matrix):
    eigvals = np.linalg.eigvals(covariance_matrix).real
    eigvals = np.maximum(eigvals, 1e-12) # clip negative/zeros
    p = eigvals / np.sum(eigvals)
    entropy = -np.sum(p * np.log(p))
    er = np.exp(entropy)
    pr = (np.sum(eigvals))**2 / np.sum(eigvals**2)
    return eigvals, er, pr

def main():
    print("="*70)
    print("   FAST EMBEDDING DIAGNOSTICS SUITE")
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
    all_subject_data = loader.load_all()

    # Extract Evaluation Windows
    print("\nExtracting Evaluation Windows...")
    eval_windows = extract_evaluation_windows(all_subject_data, test_subs)
    print(f"Total Evaluation Windows: {len(eval_windows)}")

    model = ContrastiveAADModel().to(device)
    
    # -------------------------------------------------------------
    # EPOCH 0 (Random Baseline)
    # -------------------------------------------------------------
    print("\n--- Running Epoch 0 Diagnostics ---")
    E0, A0, B0 = get_embeddings(model, eval_windows, device)
    
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
    # EPOCH 5 (Trained Model)
    # -------------------------------------------------------------
    print("\n--- Running Epoch 5 Diagnostics ---")
    E5, A5, B5 = get_embeddings(model, eval_windows, device)
    
    # =============================================================
    # DIAGNOSTICS COMPUTATION & PLOTTING
    # =============================================================
    print("\nGenerating Diagnostic Artifacts...")
    
    metadata = pd.DataFrame([{
        "subject": w["subject"],
        "story": w["attended_story"],
        "ear": w["attended_ear"]
    } for w in eval_windows])

    stats = []
    retrieval_res = []

    for epoch_name, E, A, B in [("epoch0", E0, A0, B0), ("epoch5", E5, A5, B5)]:
        # Diagnostic 1: Similarity Distributions
        cos_att = (E * A).sum(axis=-1)
        cos_unatt = (E * B).sum(axis=-1)
        margin = cos_att - cos_unatt
        
        plt.figure(figsize=(8,6))
        sns.histplot(cos_att, color="blue", label="Attended", alpha=0.5, kde=True)
        sns.histplot(cos_unatt, color="red", label="Unattended", alpha=0.5, kde=True)
        plt.legend()
        plt.title(f"Cosine Similarity ({epoch_name})")
        plt.savefig(out_dir / f"similarity_histogram_{epoch_name}.png")
        plt.close()
        
        plt.figure(figsize=(8,6))
        sns.histplot(margin, color="green", kde=True)
        plt.axvline(0, color='black', linestyle='--')
        plt.title(f"Margin (Attended - Unattended) ({epoch_name})")
        plt.savefig(out_dir / f"margin_histogram_{epoch_name}.png")
        plt.close()
        
        stats.append({
            "Epoch": epoch_name,
            "Mean_Cos_Att": cos_att.mean(),
            "Mean_Cos_Unatt": cos_unatt.mean(),
            "Mean_Margin": margin.mean(),
            "Std_Margin": margin.std()
        })
        
        # Diagnostic 2: Retrieval
        r1, r5, med_r, mean_r = compute_retrieval_metrics(E, A)
        retrieval_res.append({
            "Epoch": epoch_name,
            "Recall@1": r1,
            "Recall@5": r5,
            "Median_Rank": med_r,
            "Mean_Rank": mean_r
        })
        
        # Diagnostic 3: Geometry
        cov = np.cov(E.T)
        eigvals, er, pr = effective_rank(cov)
        
        plt.figure(figsize=(8,6))
        plt.plot(sorted(eigvals, reverse=True))
        plt.yscale('log')
        plt.title(f"EEG Embedding Spectrum ({epoch_name})\nEff Rank: {er:.1f} | Part. Ratio: {pr:.1f}")
        plt.savefig(out_dir / f"embedding_spectrum_{epoch_name}.png")
        plt.close()
        
        stats[-1]["Effective_Rank"] = er
        stats[-1]["Participation_Ratio"] = pr

        # Diagnostic 4: PCA
        pca = PCA(n_components=2)
        E_pca = pca.fit_transform(E)
        
        for hue_col in ["subject", "story", "ear"]:
            plt.figure(figsize=(8,6))
            sns.scatterplot(x=E_pca[:,0], y=E_pca[:,1], hue=metadata[hue_col], palette="Set1", s=50, alpha=0.7)
            plt.title(f"PCA ({epoch_name}) colored by {hue_col}")
            plt.savefig(out_dir / f"embedding_{hue_col}_pca_{epoch_name}.png")
            plt.close()

    # Diagnostic 5: Cross-Subject Retrieval (Epoch 5 only)
    subs = test_subs
    cross_retrieval = np.zeros((len(subs), len(subs)))
    
    for i, s_query in enumerate(subs):
        idx_query = (metadata["subject"] == s_query).values
        E_query = E5[idx_query]
        for j, s_target in enumerate(subs):
            idx_target = (metadata["subject"] == s_target).values
            A_target = A5[idx_target]
            
            # Since test subjects might have different number of windows due to filtering,
            # we do standard retrieval if shapes match, or pairwise otherwise
            if len(E_query) == len(A_target):
                # For KUL, subjects heard the exact same trials, so lengths match
                r1, _, _, _ = compute_retrieval_metrics(E_query, A_target)
                cross_retrieval[i, j] = r1
            else:
                cross_retrieval[i, j] = np.nan
                
    df_cross = pd.DataFrame(cross_retrieval, index=subs, columns=subs)
    df_cross.to_csv(out_dir / "cross_subject_retrieval_matrix.csv")
    
    plt.figure(figsize=(6,5))
    sns.heatmap(df_cross, annot=True, cmap="YlGnBu", fmt=".3f")
    plt.title("Cross-Subject Retrieval (Recall@1)")
    plt.xlabel("Audio Target Subject")
    plt.ylabel("EEG Query Subject")
    plt.savefig(out_dir / "cross_subject_retrieval_heatmap.png")
    plt.close()

    # Diagnostic 6: Representation Drift
    drift_sims = (E0 * E5).sum(axis=-1)
    stats[-1]["Avg_Drift_Sim"] = drift_sims.mean()
    stats[-1]["Min_Drift_Sim"] = drift_sims.min() # Max drift is min similarity
    
    # Save statistics
    pd.DataFrame(stats).to_csv(out_dir / "embedding_statistics.csv", index=False)
    pd.DataFrame(retrieval_res).to_csv(out_dir / "retrieval_metrics.csv", index=False)

    print(f"\nAll diagnostics completed. Saved to {out_dir}")

if __name__ == "__main__":
    main()
