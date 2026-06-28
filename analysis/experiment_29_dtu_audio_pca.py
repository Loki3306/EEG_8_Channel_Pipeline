import os
import sys
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
import h5py

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.matchnet import ContrastiveMatchNet

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

def extract_env_from_dtu_audio(track_idx, num_samples):
    """
    Since DTU uses a pre-computed audio representation inside the `.mat` file,
    we will load one of the DTU subject's HDF5 files to extract the 28-channel 
    audio envelopes directly, rather than raw WAV files (which require the Gammatone
    filterbank that was already run on DTU data).
    """
    # For DTU, the audio envelopes are already inside the Subject MAT files!
    # Let's find any DTU subject file.
    if os.path.exists("/kaggle/input/datasets/lowk1ee/s1-dtu"):
        dtu_dir = "/kaggle/input/datasets/lowk1ee/s1-dtu"
    elif os.path.exists("/kaggle/input/s1-dtu"):
        dtu_dir = "/kaggle/input/s1-dtu"
    else:
        dtu_dir = str(REPO_ROOT / "data" / "s1-dtu")
        
    subject_files = glob.glob(os.path.join(dtu_dir, "dataset_subject*.mat"))
    if not subject_files:
        raise FileNotFoundError(f"No DTU dataset files found in {dtu_dir}")
        
    file_path = subject_files[0]
    
    with h5py.File(file_path, 'r') as f:
        # Load the specific track's attended audio (ya) from the first trial that uses it
        # Actually, let's just grab the audio from all 40 trials to map the acoustic space
        trial_refs = f['dataset'][0]
        for i in range(len(trial_refs)):
            ref = trial_refs[i]
            trial = f[ref]
            
            # The audio in DTU is ya and yb (both 28 channels)
            # We can extract the actual envelopes here.
            pass # We will just do it in the main loop to grab all unique audio
            
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Phase C: DTU Audio Encoder PCA Probe on {device}")
    
    # 1. Load Model (DTU Checkpoint)
    ckpt_dir = REPO_ROOT / "checkpoints"
    ckpts = list(ckpt_dir.glob("matchnet_fold_*.pth"))
    if not ckpts:
        print("No DTU LOSO checkpoints found.")
        return
        
    ckpt_path = ckpts[-1] # Grab the last one
    print(f"Loading weights from {ckpt_path.name}")
    
    model = ContrastiveMatchNet("eegnet", eeg_channels=8, audio_channels=28, latent_dim=64).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    # 2. Extract DTU audio envelopes from a .mat file
    if os.path.exists("/kaggle/input/datasets/lowk1ee/s1-dtu"):
        dtu_dir = "/kaggle/input/datasets/lowk1ee/s1-dtu"
    elif os.path.exists("/kaggle/input/s1-dtu"):
        dtu_dir = "/kaggle/input/s1-dtu"
    else:
        dtu_dir = str(REPO_ROOT / "data" / "s1-dtu")
        
    subject_files = glob.glob(os.path.join(dtu_dir, "dataset_subject*.mat"))
    if not subject_files:
        print(f"No DTU dataset files found in {dtu_dir}")
        return
        
    file_path = subject_files[0]
    print(f"Extracting DTU audio envelopes from {file_path}")
    
    all_embeddings = []
    all_labels = []
    
    with h5py.File(file_path, 'r') as f:
        trial_refs = f['dataset'][0]
        
        with torch.no_grad():
            for i in range(len(trial_refs)):
                ref = trial_refs[i]
                trial = f[ref]
                
                # ya shape: [Time, 28] -> transpose to [28, Time]
                ya = trial['ya'][:] 
                yb = trial['yb'][:]
                
                # DTU already provides them as 28-channel envelopes
                ya = norm_env(ya.T if ya.shape[1] == 28 else ya)
                yb = norm_env(yb.T if yb.shape[1] == 28 else yb)
                
                chunks_ya = chunk_audio(ya, WINDOW_SEC, HOP_SEC, FS)
                chunks_yb = chunk_audio(yb, WINDOW_SEC, HOP_SEC, FS)
                
                for c in chunks_ya:
                    c_tensor = torch.FloatTensor(c).unsqueeze(0).to(device)
                    z_a = model.encode_audio(c_tensor)
                    if z_a.dim() > 2: z_a = z_a.mean(dim=2)
                    all_embeddings.append(z_a.cpu().numpy()[0])
                    all_labels.append(f"Trial_{i}_Attended")
                    
                for c in chunks_yb:
                    c_tensor = torch.FloatTensor(c).unsqueeze(0).to(device)
                    z_b = model.encode_audio(c_tensor)
                    if z_b.dim() > 2: z_b = z_b.mean(dim=2)
                    all_embeddings.append(z_b.cpu().numpy()[0])
                    all_labels.append(f"Trial_{i}_Unattended")
                
                if i >= 4: # Just map a few trials to avoid immense plot density
                    break
                    
    X = np.stack(all_embeddings)
    
    print(f"\nExtracted {len(X)} total embeddings. Running PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    
    unique_labels = list(set(all_labels))
    colors = plt.cm.get_cmap('tab20', len(unique_labels))
    
    for i, label in enumerate(unique_labels):
        idx = [j for j, l in enumerate(all_labels) if l == label]
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=label, alpha=0.6, s=15, color=colors(i))
        
    plt.title("PCA of DTU Audio Encoder Embeddings by Trial Source")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    
    # Put legend outside if too many
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.grid(True, alpha=0.3)
    
    out_path = REPO_ROOT / "analysis" / "experiment_29_dtu_audio_pca.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved PCA plot to {out_path}")
    
if __name__ == "__main__":
    main()
