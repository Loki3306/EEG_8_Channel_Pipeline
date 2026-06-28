import os
import sys
import re
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.matchnet import ContrastiveMatchNet
from data.extract_gammatone_envelopes import extract_gammatone_envelopes

FS = 64
WINDOW_SEC = 10
HOP_SEC = 2

def norm_env(env):
    env = env.T
    env = env - env.mean(axis=0, keepdims=True)
    env = env / (env.std(axis=0, keepdims=True) + 1e-12)
    return env.T

def chunk_audio(ya, window_sec, hop_sec, fs=FS):
    win_samples = int(window_sec * fs)
    hop_samples = int(hop_sec * fs)
    chunks_ya = []
    start = 0
    while start + win_samples <= ya.shape[1]:
        end = start + win_samples
        chunks_ya.append(ya[:, start:end])
        start += hop_samples
    return chunks_ya

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Phase 3: Audio Encoder PCA Probe on {device}")
    
    # 1. Load Model
    ckpt_dir = REPO_ROOT / "checkpoints"
    ckpts = list(ckpt_dir.glob("matchnet_kul_fold_S*_best.pth"))
    if not ckpts:
        print("No KUL LOSO checkpoints found.")
        return
        
    ckpt_path = ckpts[0]
    print(f"Loading weights from {ckpt_path.name}")
    
    model = ContrastiveMatchNet("eegnet", eeg_channels=8, audio_channels=28, latent_dim=64).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    # 2. Find audio files
    if os.path.exists("/kaggle/input/datasets/lowk1ee/audio-klu/stimuli"):
        stimuli_dir = "/kaggle/input/datasets/lowk1ee/audio-klu/stimuli"
    elif os.path.exists("/kaggle/input/audio-klu/stimuli"):
        stimuli_dir = "/kaggle/input/audio-klu/stimuli"
    else:
        stimuli_dir = str(REPO_ROOT / "data" / "audio-klu" / "stimuli")
        
    wav_files = glob.glob(os.path.join(stimuli_dir, "*_dry.wav"))
    if not wav_files:
        print(f"No _dry.wav files found in {stimuli_dir}")
        return
        
    print(f"Found {len(wav_files)} dry audio tracks.")
    
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for wav_path in wav_files:
            wav_name = os.path.basename(wav_path)
            
            # Determine which track this belongs to
            track_match = re.search(r"track(\d+)", wav_name, re.IGNORECASE)
            track_id = f"Track {track_match.group(1)}" if track_match else "Unknown"
            
            print(f"Processing {wav_name} -> {track_id}")
            env = extract_gammatone_envelopes(wav_path, target_fs=FS)
            env = norm_env(env)
            
            chunks = chunk_audio(env, WINDOW_SEC, HOP_SEC, FS)
            
            for c in chunks:
                c_tensor = torch.FloatTensor(c).unsqueeze(0).to(device)
                
                # Pass through audio encoder
                _, z_a = model.audio_net(c_tensor)
                
                # Flatten or pool depending on architecture output
                if z_a.dim() > 2:
                    z_a = z_a.mean(dim=2) # global average pool across time if it's temporal
                    
                # The projection head in MatchNet is linear for z_a
                z_a = model.proj_a(z_a)
                
                all_embeddings.append(z_a.cpu().numpy()[0])
                all_labels.append(track_id)
                
    X = np.stack(all_embeddings)
    
    print(f"\nExtracted {len(X)} total embeddings. Running PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    
    unique_labels = list(set(all_labels))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for i, label in enumerate(unique_labels):
        idx = [j for j, l in enumerate(all_labels) if l == label]
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=label, alpha=0.6, s=15, c=colors[i % len(colors)])
        
    plt.title("PCA of KUL Audio Encoder Embeddings by Story Identity")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path = REPO_ROOT / "analysis" / "experiment_25_audio_pca.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved PCA plot to {out_path}")
    
if __name__ == "__main__":
    main()
