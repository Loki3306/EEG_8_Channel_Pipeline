import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import torch.nn.functional as F
import glob

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from baselines.ridge_aad import load_subject_examples, subject_files
from training.train_matchnet_loso import prepare_dataset, get_mapping_data
from models.matchnet import ContrastiveMatchNet

FS = 64
CHANNELS = [13, 46, 43, 23, 50, 0, 52, 14]
LOWCUT = 1.0
HIGHCUT = 6.0
NUM_BANDS = 28
WINDOW_SEC = 2.0
BASE_HOP_SEC = 0.25 # Extract at finest granularity

def evaluate_trial_similarities(model, x_np, ya_np, yb_np, device, batch_size=256):
    """
    Evaluate a single full-length trial and return similarities at 0.25s intervals.
    Uses batched inference to maximize GPU utilization.
    """
    window_samples = int(WINDOW_SEC * FS)
    hop_samples = int(BASE_HOP_SEC * FS)
    
    x_chunks, ya_chunks, yb_chunks = [], [], []
    
    start = 0
    while start + window_samples <= x_np.shape[1]:
        end = start + window_samples
        x_chunks.append(x_np[:, start:end])
        ya_chunks.append(ya_np[:, start:end])
        yb_chunks.append(yb_np[:, start:end])
        start += hop_samples
        
    if not x_chunks:
        return np.array([]), np.array([])
        
    x_tensor = torch.FloatTensor(np.stack(x_chunks))
    ya_tensor = torch.FloatTensor(np.stack(ya_chunks))
    yb_tensor = torch.FloatTensor(np.stack(yb_chunks))
    
    dataset = torch.utils.data.TensorDataset(x_tensor, ya_tensor, yb_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    sims_a, sims_b = [], []
    
    with torch.no_grad():
        for bx, bya, byb in loader:
            bx, bya, byb = bx.to(device), bya.to(device), byb.to(device)
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                z_eeg, z_a, z_b = model(bx, bya, byb)
                # Cosine similarity across channel dimension (dim=1) yields [B, T]
                # Mean across time dimension (dim=1) yields [B]
                batch_sim_a = F.cosine_similarity(z_eeg, z_a, dim=1).mean(dim=1).cpu().numpy()
                batch_sim_b = F.cosine_similarity(z_eeg, z_b, dim=1).mean(dim=1).cpu().numpy()
            
            sims_a.extend(batch_sim_a)
            sims_b.extend(batch_sim_b)
        
    return np.array(sims_a), np.array(sims_b)

def aggregate_decisions(sims_a, sims_b, hop_sec, method="logit"):
    """
    Subsample similarities according to hop_sec and aggregate over time.
    Returns array of cumulative correctness (1 or 0) for each step.
    """
    # Downsample factor from 0.25s base hop
    step = int(hop_sec / BASE_HOP_SEC)
    
    sa = sims_a[::step]
    sb = sims_b[::step]
    
    logits = sa - sb
    
    if method == "majority":
        votes = np.sign(logits)
        # Handle zero-ties by making them strictly incorrect for sum
        votes[votes == 0] = -1 
        cum_scores = np.cumsum(votes)
    else: # logit
        cum_scores = np.cumsum(logits)
        
    return cum_scores > 0

def run_study():
    os.makedirs(os.path.join(REPO_ROOT, "results"), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}...")
    
    mapping, envelopes = get_mapping_data()
    all_paths = subject_files()
    
    # Find all available checkpoints
    ckpt_files = glob.glob(str(REPO_ROOT / "checkpoints" / "matchnet_fold_S*_best.pth"))
    
    results_data = []
    
    for ckpt in ckpt_files:
        subj = os.path.basename(ckpt).split('_')[2]
        print(f"\nProcessing {subj}...")
        
        test_path = next((p for p in all_paths if p.stem.split('_')[0] == subj), None)
        if not test_path:
            continue
            
        test_exs = load_subject_examples(test_path)
        X_te, YA_te, YB_te = prepare_dataset(test_exs, CHANNELS, LOWCUT, HIGHCUT, test_path.stem, mapping, envelopes)
        
        model = ContrastiveMatchNet(eeg_model_type="eegnet", eeg_channels=len(CHANNELS), audio_channels=NUM_BANDS, latent_dim=64, audio_model_type="standard").to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        model.eval()
        
        all_sims_a = []
        all_sims_b = []
        
        # 1. Extract base similarities
        for i in range(len(X_te)):
            sa, sb = evaluate_trial_similarities(model, X_te[i], YA_te[i], YB_te[i], device)
            all_sims_a.append(sa)
            all_sims_b.append(sb)
            
        # 2. Run Experiments across different hop sizes and methods
        hop_sizes = [2.0, 1.0, 0.5, 0.25]
        methods = ["logit", "majority"]
        
        for hop in hop_sizes:
            for method in methods:
                # Time array corresponds to cumulative decisions
                # 1 decision = 2s latency. Each additional decision adds `hop` seconds latency.
                
                max_len = max(len(sa[::int(hop/BASE_HOP_SEC)]) for sa in all_sims_a)
                correct_matrix = np.zeros((len(all_sims_a), max_len))
                active_matrix = np.zeros((len(all_sims_a), max_len))
                
                for idx, (sa, sb) in enumerate(zip(all_sims_a, all_sims_b)):
                    correct = aggregate_decisions(sa, sb, hop, method)
                    L = len(correct)
                    correct_matrix[idx, :L] = correct
                    active_matrix[idx, :L] = 1
                    
                # Accuracy at each cumulative step
                with np.errstate(invalid='ignore'):
                    step_accs = np.sum(correct_matrix, axis=0) / np.sum(active_matrix, axis=0)
                
                for step_idx, acc in enumerate(step_accs):
                    if np.isnan(acc): break
                    latency = 2.0 + (step_idx * hop)
                    # For comparison with requested decision counts, specifically non-overlapping
                    if hop == 2.0:
                        decisions = step_idx + 1
                    else:
                        decisions = np.nan
                        
                    results_data.append({
                        "Subject": subj,
                        "Hop": hop,
                        "Method": method,
                        "Step": step_idx + 1,
                        "Decisions_NonOverlap": decisions,
                        "Latency_s": latency,
                        "Accuracy": acc
                    })
                    
    df = pd.DataFrame(results_data)
    csv_path = os.path.join(REPO_ROOT, "results", "evidence_accumulation.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")
    
    generate_markdown_and_plots(df)

def generate_markdown_and_plots(df):
    results_dir = os.path.join(REPO_ROOT, "results")
    
    plt.figure(figsize=(15, 5))
    
    # 1. Cumulative Voting (Non-overlapping, logit)
    plt.subplot(1, 3, 1)
    df_nonov = df[(df['Hop'] == 2.0) & (df['Method'] == 'logit')].copy()
    targets = [1, 2, 3, 5, 10, 20, 30]
    
    mean_accs = []
    for t in targets:
        v = df_nonov[df_nonov['Step'] == t]['Accuracy'].mean() * 100
        mean_accs.append(v)
        
    for subj in df_nonov['Subject'].unique():
        subj_data = df_nonov[df_nonov['Subject'] == subj]
        plt.plot(subj_data['Latency_s'], subj_data['Accuracy']*100, alpha=0.3, color='gray')
        
    mean_curve = df_nonov.groupby('Latency_s')['Accuracy'].mean().reset_index()
    plt.plot(mean_curve['Latency_s'], mean_curve['Accuracy']*100, color='blue', linewidth=2, label='Mean Accuracy')
    
    plt.title("Exp 1: Cumulative Voting (Hop=2.0s)")
    plt.xlabel("Latency (seconds)")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()
    
    # 2. Overlapping Windows
    plt.subplot(1, 3, 2)
    df_overlap = df[df['Method'] == 'logit']
    for hop in sorted(df['Hop'].unique(), reverse=True):
        curve = df_overlap[df_overlap['Hop'] == hop].groupby('Latency_s')['Accuracy'].mean().reset_index()
        # Crop to 60s for clean viewing
        curve = curve[curve['Latency_s'] <= 60]
        plt.plot(curve['Latency_s'], curve['Accuracy']*100, label=f'Hop={hop}s', linewidth=2)
        
    plt.title("Exp 2: Overlapping Windows")
    plt.xlabel("Latency (seconds)")
    plt.grid(True)
    plt.legend()
    
    # 3. Log-Odds vs Majority
    plt.subplot(1, 3, 3)
    for method in ['logit', 'majority']:
        curve = df[(df['Hop'] == 2.0) & (df['Method'] == method)].groupby('Latency_s')['Accuracy'].mean().reset_index()
        curve = curve[curve['Latency_s'] <= 60]
        plt.plot(curve['Latency_s'], curve['Accuracy']*100, label=f'Method={method}', linewidth=2)
        
    plt.title("Exp 3: Log-Odds vs Majority (Hop=2.0s)")
    plt.xlabel("Latency (seconds)")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    img_path = os.path.join(results_dir, "evidence_accumulation.png")
    plt.savefig(img_path)
    print(f"Saved {img_path}")
    
    # Generate Markdown
    md_path = os.path.join(results_dir, "evidence_accumulation.md")
    with open(md_path, 'w') as f:
        f.write("# Evidence Accumulation Study\n\n")
        
        f.write("## Exp 1: Cumulative Voting (Hop = 2.0s)\n")
        f.write("| Decisions | Latency (s) | Mean Accuracy |\n")
        f.write("| --------- | ----------- | ------------- |\n")
        for t, m in zip(targets, mean_accs):
            if not np.isnan(m):
                f.write(f"| {t} | {t*2}s | {m:.2f}% |\n")
                
        f.write("\n## Exp 2: Overlapping Windows (Final Accuracy at 60s)\n")
        f.write("| Hop Size | Final Mean Accuracy |\n")
        f.write("| -------- | ------------------- |\n")
        for hop in sorted(df['Hop'].unique(), reverse=True):
            val = df_overlap[(df_overlap['Hop'] == hop) & (df_overlap['Latency_s'] <= 60)]['Accuracy'].values
            if len(val) > 0:
                f.write(f"| {hop}s | {val[-1]*100:.2f}% |\n")
                
        f.write("\n## Exp 3: Aggregation Method (Hop = 2.0s at 60s)\n")
        f.write("| Method | Final Mean Accuracy |\n")
        f.write("| ------ | ------------------- |\n")
        for method in ['logit', 'majority']:
            val = df[(df['Hop'] == 2.0) & (df['Method'] == method) & (df['Latency_s'] <= 60)]['Accuracy'].values
            if len(val) > 0:
                f.write(f"| {method} | {val[-1]*100:.2f}% |\n")
                
    print(f"Saved {md_path}")

if __name__ == "__main__":
    run_study()
