import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import scipy.cluster.hierarchy as sch
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.kul_cached_dataset import KULCachedLoader

def safe_corr_np(x, y, eps=1e-8):
    x_mean = x.mean()
    y_mean = y.mean()
    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
    return num / (den + eps)

def evaluate_trial(pred, wav_a, wav_b, window_sec=10, hop_sec=1.0, fs=64):
    num_bands = pred.shape[0]
    win_samples = int(window_sec * fs)
    hop_samples = int(hop_sec * fs)
    
    if win_samples >= pred.shape[1]:
        c_a = np.mean([safe_corr_np(pred[i], wav_a[i]) for i in range(num_bands)])
        c_b = np.mean([safe_corr_np(pred[i], wav_b[i]) for i in range(num_bands)])
        return c_a > c_b, 1, 1 if c_a > c_b else 0
        
    correct_windows = 0
    total_windows = 0
    
    for start in range(0, pred.shape[1] - win_samples + 1, hop_samples):
        stop = start + win_samples
        c_a = np.mean([safe_corr_np(pred[i, start:stop], wav_a[i, start:stop]) for i in range(num_bands)])
        c_b = np.mean([safe_corr_np(pred[i, start:stop], wav_b[i, start:stop]) for i in range(num_bands)])
        if c_a > c_b:
            correct_windows += 1
        total_windows += 1
        
    if total_windows == 0:
        return False, 0, 0
        
    trial_correct = (correct_windows > total_windows / 2.0)
    return trial_correct, total_windows, correct_windows

def main():
    print("================================================================")
    print("           PER-SUBJECT RIDGE CEILING (ORACLE ANALYSIS)          ")
    print("================================================================")
    
    loader = KULCachedLoader(REPO_ROOT / "data" / "processed_kul")
    try:
        all_subject_data = loader.load_all()
    except FileNotFoundError:
        print("KUL Cache not found. Please run preprocessing first.")
        return
        
    out_dir = REPO_ROOT / "results" / "subject_oracle"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    num_channels = 8
    num_lags = 17 
    feature_count = num_channels * num_lags
    num_bands = 28
    
    ridge_lambda = 100.0
    
    # Store XTX and XTY for everything
    trial_xtx = defaultdict(list)
    trial_xty = defaultdict(list)
    
    subject_xtx = {}
    subject_xty = {}
    
    global_xtx = np.zeros((feature_count, feature_count), dtype=float)
    global_xty = np.zeros((feature_count, num_bands), dtype=float)
    
    print("\nPhase 1: Accumulating Matrices per Trial, Subject, and Global...")
    for sub, trials in all_subject_data.items():
        s_xtx = np.zeros((feature_count, feature_count), dtype=float)
        s_xty = np.zeros((feature_count, num_bands), dtype=float)
        
        for t in trials:
            eeg_np = t["eeg"].numpy()
            a_np = t["audio_a"].numpy()
            
            e_mean = eeg_np.mean(axis=1, keepdims=True)
            e_std = eeg_np.std(axis=1, keepdims=True) + 1e-12
            e_norm = (eeg_np - e_mean) / e_std
            
            a_mean = a_np.mean(axis=1, keepdims=True)
            a_std = a_np.std(axis=1, keepdims=True) + 1e-12
            a_norm = (a_np - a_mean) / a_std
            
            lagged_blocks = []
            for lag in range(num_lags):
                if lag == 0:
                    lagged_blocks.append(e_norm.T)
                else:
                    shifted = np.vstack([np.zeros((lag, num_channels)), e_norm.T[:-lag]])
                    lagged_blocks.append(shifted)
                    
            X_mat = np.concatenate(lagged_blocks, axis=1) # (Time, Channels*Lags)
            Y_mat = a_norm.T # (Time, 28)
            
            t_xtx = X_mat.T @ X_mat
            t_xty = X_mat.T @ Y_mat
            
            trial_xtx[sub].append(t_xtx)
            trial_xty[sub].append(t_xty)
            
            s_xtx += t_xtx
            s_xty += t_xty
            
        subject_xtx[sub] = s_xtx
        subject_xty[sub] = s_xty
        
        global_xtx += s_xtx
        global_xty += s_xty
        
    print("\nPhase 2: Evaluating Oracle (LOTO) vs Global (LOSO)...")
    
    loso_metrics = defaultdict(lambda: {"trials_ok": 0, "win_ok": 0, "win_tot": 0, "margins": []})
    oracle_metrics = defaultdict(lambda: {"trials_ok": 0, "win_ok": 0, "win_tot": 0, "margins": []})
    
    subject_decoders = {} # Dictionary to store fully-trained Subject Decoders
    
    for sub in all_subject_data.keys():
        # --- LOSO Decoder ---
        loso_train_xtx = global_xtx - subject_xtx[sub]
        loso_train_xty = global_xty - subject_xty[sub]
        W_loso = np.linalg.solve(loso_train_xtx + ridge_lambda * np.eye(feature_count), loso_train_xty)
        
        # --- Fully-trained Subject Decoder (for Clustering/PCA later) ---
        W_sub = np.linalg.solve(subject_xtx[sub] + ridge_lambda * np.eye(feature_count), subject_xty[sub])
        subject_decoders[sub] = W_sub.flatten()
        
        trials = all_subject_data[sub]
        for idx, t in enumerate(trials):
            # --- Oracle (LOTO) Decoder ---
            # Train on all trials of this subject EXCEPT the current one
            if len(trials) > 1:
                oracle_train_xtx = subject_xtx[sub] - trial_xtx[sub][idx]
                oracle_train_xty = subject_xty[sub] - trial_xty[sub][idx]
                W_oracle = np.linalg.solve(oracle_train_xtx + ridge_lambda * np.eye(feature_count), oracle_train_xty)
            else:
                # Fallback if subject only has 1 trial (not applicable to KUL, but safe)
                W_oracle = W_sub 
                
            # Data for inference
            eeg_np = t["eeg"].numpy()
            audio_a = t["audio_a"].numpy()
            audio_b = t["audio_b"].numpy()
            
            e_mean = eeg_np.mean(axis=1, keepdims=True)
            e_std = eeg_np.std(axis=1, keepdims=True) + 1e-12
            e_norm = (eeg_np - e_mean) / e_std
            
            a_mean = audio_a.mean(axis=1, keepdims=True)
            a_std = audio_a.std(axis=1, keepdims=True) + 1e-12
            a_norm = (audio_a - a_mean) / a_std
            
            b_mean = audio_b.mean(axis=1, keepdims=True)
            b_std = audio_b.std(axis=1, keepdims=True) + 1e-12
            b_norm = (audio_b - b_mean) / b_std
            
            lagged_blocks = []
            for lag in range(num_lags):
                if lag == 0:
                    lagged_blocks.append(e_norm.T)
                else:
                    shifted = np.vstack([np.zeros((lag, num_channels)), e_norm.T[:-lag]])
                    lagged_blocks.append(shifted)
            X_mat = np.concatenate(lagged_blocks, axis=1)
            
            # Predict LOSO
            pred_loso = (X_mat @ W_loso).T
            c_att_loso = np.mean([safe_corr_np(pred_loso[i], a_norm[i]) for i in range(num_bands)])
            c_unatt_loso = np.mean([safe_corr_np(pred_loso[i], b_norm[i]) for i in range(num_bands)])
            loso_margin = c_att_loso - c_unatt_loso
            t_ok_l, n_win_l, c_win_l = evaluate_trial(pred_loso, a_norm, b_norm, window_sec=10, hop_sec=1.0)
            
            loso_metrics[sub]["trials_ok"] += int(t_ok_l)
            loso_metrics[sub]["win_ok"] += c_win_l
            loso_metrics[sub]["win_tot"] += n_win_l
            loso_metrics[sub]["margins"].append(loso_margin)
            
            # Predict Oracle
            pred_oracle = (X_mat @ W_oracle).T
            c_att_oracle = np.mean([safe_corr_np(pred_oracle[i], a_norm[i]) for i in range(num_bands)])
            c_unatt_oracle = np.mean([safe_corr_np(pred_oracle[i], b_norm[i]) for i in range(num_bands)])
            oracle_margin = c_att_oracle - c_unatt_oracle
            t_ok_o, n_win_o, c_win_o = evaluate_trial(pred_oracle, a_norm, b_norm, window_sec=10, hop_sec=1.0)
            
            oracle_metrics[sub]["trials_ok"] += int(t_ok_o)
            oracle_metrics[sub]["win_ok"] += c_win_o
            oracle_metrics[sub]["win_tot"] += n_win_o
            oracle_metrics[sub]["margins"].append(oracle_margin)

    print("\n================================================================")
    print("RESULTS: ORACLE (LOTO) vs GLOBAL (LOSO)")
    print("================================================================")
    print(f"{'Subject':<8} | {'LOSO Acc':<10} | {'Oracle Acc':<12} | {'Gap (Acc)':<10} | {'LOSO Marg':<10} | {'Oracle Marg':<12} | {'Gap (Marg)'}")
    print("-" * 90)
    
    subs = sorted(list(all_subject_data.keys()), key=lambda x: int(x[1:]))
    
    loso_accs = []
    oracle_accs = []
    
    for sub in subs:
        l_m = loso_metrics[sub]
        o_m = oracle_metrics[sub]
        
        l_acc = l_m["trials_ok"] / len(l_m["margins"])
        o_acc = o_m["trials_ok"] / len(o_m["margins"])
        gap_acc = o_acc - l_acc
        
        loso_accs.append(l_acc * 100)
        oracle_accs.append(o_acc * 100)
        
        l_marg = np.median(l_m["margins"])
        o_marg = np.median(o_m["margins"])
        gap_marg = o_marg - l_marg
        
        print(f"{sub:<8} | {l_acc*100:>9.1f}% | {o_acc*100:>11.1f}% | {gap_acc*100:>+9.1f}% | {l_marg:>10.4f} | {o_marg:>11.4f} | {gap_marg:>+10.4f}")

    print("-" * 90)
    print(f"MEDIAN   | {np.median(loso_accs):>9.1f}% | {np.median(oracle_accs):>11.1f}% | {(np.median(oracle_accs)-np.median(loso_accs)):>+9.1f}%")
    print(f"MEAN     | {np.mean(loso_accs):>9.1f}% | {np.mean(oracle_accs):>11.1f}% | {(np.mean(oracle_accs)-np.mean(loso_accs)):>+9.1f}%")

    # Plot Comparison
    plt.figure(figsize=(12, 6))
    x = np.arange(len(subs))
    width = 0.35
    plt.bar(x - width/2, loso_accs, width, label='LOSO (Global Model)', color='royalblue')
    plt.bar(x + width/2, oracle_accs, width, label='Oracle (Subject-Specific)', color='forestgreen')
    plt.axhline(y=50, color='r', linestyle='--', label='Chance (50%)')
    plt.ylabel('Trial Accuracy (%)')
    plt.title('Global vs Subject-Specific Ridge Decoders (KUL)')
    plt.xticks(x, subs)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "oracle_vs_loso_comparison.png")
    plt.close()

    print("\nPhase 3 & 4: Decoder Extraction, Clustering, and PCA...")
    decoder_matrix = []
    for sub in subs:
        decoder_matrix.append(subject_decoders[sub])
    decoder_matrix = np.array(decoder_matrix) # (16, Features)
    
    # Standardize decoders for PCA to focus on structural variance, not magnitude
    dec_mean = decoder_matrix.mean(axis=0)
    dec_std = decoder_matrix.std(axis=0) + 1e-8
    decoder_norm = (decoder_matrix - dec_mean) / dec_std
    
    # 1. Cosine Similarity Heatmap (Cosine is naturally magnitude invariant)
    sim_matrix = cosine_similarity(decoder_matrix)
    
    plt.figure(figsize=(10, 8))
    sns.clustermap(
        sim_matrix,
        xticklabels=subs,
        yticklabels=subs,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        figsize=(10, 8)
    )
    plt.title("Hierarchical Clustering of Subject-Specific Decoders (Cosine Sim)")
    plt.savefig(out_dir / "decoder_similarity_clustermap.png")
    plt.close()
    
    # 2. PCA Explained Variance
    pca = PCA()
    pca.fit(decoder_norm)
    exp_var_cum = np.cumsum(pca.explained_variance_ratio_)
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(exp_var_cum) + 1), exp_var_cum, marker='o', linestyle='-', color='purple')
    plt.axhline(y=0.90, color='r', linestyle='--', label='90% Variance Explained')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA of Subject-Specific Decoders')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "decoder_pca_explained_variance.png")
    plt.close()
    
    print("\n================================================================")
    print("PREDICTION ANSWERS (Based on Structural Variance)")
    print("================================================================")
    
    # Number of PCs to explain 90% variance
    n_pcs_90 = np.argmax(exp_var_cum >= 0.90) + 1
    
    print("1. Does the global decoder underperform because subjects require different decoders?")
    median_gap = np.median(oracle_accs) - np.median(loso_accs)
    if median_gap > 5.0:
        print(f"   -> YES. The Oracle (subject-specific) model outperforms the Global model by a median of +{median_gap:.1f}%.")
        print("      This strongly indicates that forcing a universal decoder suppresses performance.")
    else:
        print(f"   -> NO. The Oracle model only improves upon the Global model by a median of +{median_gap:.1f}%.")
        print("      This suggests that subject adaptation alone cannot fix the fundamental information bottleneck.")
        
    print("\n2. How much accuracy is theoretically recoverable from subject-specific adaptation?")
    print(f"   -> The ceiling rises from {np.median(loso_accs):.1f}% (Global) to {np.median(oracle_accs):.1f}% (Oracle).")
    
    print("\n3. Is there one universal decoder or several distinct families?")
    print(f"   -> It requires {n_pcs_90} Principal Components to explain 90% of the structural variance among the 16 subject decoders.")
    if n_pcs_90 <= 3:
        print("      This suggests decoders form a tight cluster (a nearly universal decoder family).")
    else:
        print("      This suggests vast inter-subject variability, requiring multiple distinct decoder manifolds.")
        
    print("\nAudit complete! Plots saved to results/subject_oracle/")

if __name__ == "__main__":
    main()
