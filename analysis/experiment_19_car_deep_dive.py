import os
import sys
import math
import numpy as np
import scipy.io
import scipy.signal
import scipy.linalg
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.experiment_16_input_equivalence import get_dtu_tensor
from analysis.experiment_18_car_ablation import FullLayerProfiler, compute_frechet_distance, compute_cosine_similarity
from models.matchnet import ContrastiveMatchNet

def get_kul_tensor_uniform(apply_car=False, windows_per_trial=20):
    mat_path = "/kaggle/input/datasets/lowk1ee/s1-klu/S1_KLU.mat"
    if not os.path.exists(mat_path):
        mat_path = "data/S1_KLU.mat"
    if not os.path.exists(mat_path):
        print("Missing KUL data.")
        return None, None, None, None
        
    mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    trials = mat['trials'] if 'trials' in mat else mat['trial']
    e_list, a_list, u_list = [], [], []
    t_idx_list = []
    win_len = int(3 * 64)
    
    cache_path = "kul_gammatone_cache.pkl"
    if os.path.exists(cache_path):
        import pickle
        with open(cache_path, "rb") as f:
            envelope_cache = pickle.load(f)
    else:
        envelope_cache = {}
    
    for t_idx, trial in enumerate(trials):
        eeg_data = trial.RawData.EegData
        fs_eeg = trial.FileHeader.SampleRate
        channel_names = [ch.Label for ch in trial.FileHeader.Channels]
        
        target_channels = ['T7', 'C2', 'FT8', 'P7', 'CPz', 'Fp1', 'TP8', 'C3']
        
        if apply_car:
            eeg_data = eeg_data - eeg_data.mean(axis=1, keepdims=True)
            
        try:
            sel_idx = [channel_names.index(tc) if tc in channel_names else [c.upper() for c in channel_names].index(tc.upper()) for tc in target_channels]
        except ValueError as e:
            continue
            
        eeg_8 = eeg_data[:, sel_idx] # shape: (T, 8)
        
        nyq = 0.5 * fs_eeg
        b, a = scipy.signal.butter(4, [1.0/nyq, 8.0/nyq], btype='band')
        eeg_8 = scipy.signal.filtfilt(b, a, eeg_8, axis=0)
        
        g = math.gcd(64, int(fs_eeg))
        eeg_8 = scipy.signal.resample_poly(eeg_8, 64 // g, int(fs_eeg) // g, axis=0)
        
        arr = eeg_8 - eeg_8.mean(axis=0, keepdims=True)
        scale = arr.std(axis=0, keepdims=True) + 1e-12
        eeg_norm = arr / scale
        
        att_ear = trial.attended_ear
        stimuli = trial.stimuli
        if len(stimuli) < 2: continue
        att_wav_name = str(stimuli[0] if att_ear == 'L' else stimuli[1]).strip()
        unatt_wav_name = str(stimuli[1] if att_ear == 'L' else stimuli[0]).strip()
        
        def find_wav(name):
            wav_dir = "/kaggle/input/datasets/lowk1ee/audio-klu" if os.path.exists("/kaggle/input/datasets/lowk1ee/audio-klu") else "data"
            for r, d, f in os.walk(wav_dir):
                for file in f:
                    if name in file and file.endswith(".wav"):
                        return os.path.join(r, file)
            return None
            
        att_wav_path = find_wav(att_wav_name)
        unatt_wav_path = find_wav(unatt_wav_name)
        
        if att_wav_path and unatt_wav_path and att_wav_path in envelope_cache and unatt_wav_path in envelope_cache:
            env_att = envelope_cache[att_wav_path]
            env_unatt = envelope_cache[unatt_wav_path]
            
            def norm_env(env):
                env = env.T
                env = env - env.mean(axis=0, keepdims=True)
                env = env / (env.std(axis=0, keepdims=True) + 1e-12)
                return env.T
                
            env_att = norm_env(env_att)
            env_unatt = norm_env(env_unatt)
            
            min_len = min(len(eeg_norm), env_att.shape[1], env_unatt.shape[1])
            max_start = min_len - win_len
            if max_start > 0:
                starts = np.linspace(0, max_start, windows_per_trial, dtype=int)
                for start in starts:
                    e_list.append(eeg_norm[start:start+win_len].T)
                    a_list.append(env_att[:, start:start+win_len])
                    u_list.append(env_unatt[:, start:start+win_len])
                    t_idx_list.append(t_idx)
                    
    return np.array(e_list), np.array(a_list), np.array(u_list), np.array(t_idx_list)

def evaluate_extended(model, profiler, e_tensor, a_tensor, u_tensor, t_idx_list=None):
    profiler.clear()
    margins = []
    trial_margins = {}
    embeddings = []
    
    for i in range(len(e_tensor)):
        e = torch.tensor(e_tensor[i], dtype=torch.float32).unsqueeze(0)
        a = torch.tensor(a_tensor[i], dtype=torch.float32).unsqueeze(0)
        u = torch.tensor(u_tensor[i], dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            emb_e, emb_a, emb_u = model(e, a, u)
            
            sim_a = torch.nn.functional.cosine_similarity(emb_e, emb_a, dim=1).mean().item()
            sim_u = torch.nn.functional.cosine_similarity(emb_e, emb_u, dim=1).mean().item()
            margin = sim_a - sim_u
            margins.append(margin)
            
            emb_flat = emb_e.mean(dim=-1).cpu().numpy()[0]
            embeddings.append(emb_flat)
            
            if t_idx_list is not None:
                t_idx = t_idx_list[i]
                if t_idx not in trial_margins:
                    trial_margins[t_idx] = []
                trial_margins[t_idx].append(margin)
                
    acts = profiler.get_activations()
    stats = {}
    for layer, A in acts.items():
        mu = np.mean(A, axis=0)
        sigma = np.cov(A, rowvar=False)
        stats[layer] = (mu, sigma)
        
    return margins, trial_margins, stats, np.array(embeddings)

def run_experiment():
    model_path = None
    for r, d, f in os.walk("/kaggle/input"):
        for file in f:
            if file.endswith(".pth") and "matchnet" in file:
                model_path = os.path.join(r, file)
                break
        if model_path: break
        
    if not model_path:
        for r, d, f in os.walk("checkpoints"):
            for file in f:
                if file.endswith(".pth") and "matchnet" in file:
                    model_path = os.path.join(r, file)
                    break
            if model_path: break
            
    if not model_path:
        print("Could not find model.")
        return

    model = ContrastiveMatchNet(eeg_model_type="eegnet", eeg_channels=8, audio_channels=28, latent_dim=64)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    profiler = FullLayerProfiler(model)
    
    print("Loading DTU Target...")
    e_dtu, a_dtu, u_dtu = get_dtu_tensor()
    if e_dtu is not None: e_dtu, a_dtu, u_dtu = e_dtu[:400], a_dtu[:400], u_dtu[:400]
    
    _, _, stat_dtu, emb_dtu = evaluate_extended(model, profiler, e_dtu, a_dtu, u_dtu)
    
    print("Loading KUL Baseline...")
    e_base, a_base, u_base, t_base = get_kul_tensor_uniform(apply_car=False)
    m_base, tm_base, stat_base, emb_base = evaluate_extended(model, profiler, e_base, a_base, u_base, t_base)
    
    print("Loading KUL CAR...")
    e_car, a_car, u_car, t_car = get_kul_tensor_uniform(apply_car=True)
    m_car, tm_car, stat_car, emb_car = evaluate_extended(model, profiler, e_car, a_car, u_car, t_car)
    
    # 1. Per-trial Table
    print("\n================================================================================")
    print("1. PER-TRIAL CAR ANALYSIS")
    print("================================================================================")
    print(f"| {'Trial':<5} | {'Base Margin':<11} | {'CAR Margin':<10} | {'Δ Margin':<9} | {'Base Correct':<12} | {'CAR Correct':<11} |")
    print("-" * 75)
    
    all_trials = sorted(list(set(t_base) | set(t_car)))
    for t in all_trials:
        mb = np.mean(tm_base[t]) if t in tm_base else 0
        mc = np.mean(tm_car[t]) if t in tm_car else 0
        delta = mc - mb
        bc = "Yes" if mb > 0 else "No"
        cc = "Yes" if mc > 0 else "No"
        print(f"| {t:<5} | {mb:>11.4f} | {mc:>10.4f} | {delta:>9.4f} | {bc:>12} | {cc:>11} |")
        
    # 2. Layer Trajectory Plot
    layers = ["SpatialConv", "BN2", "Block1_Out", "Block2_Out", "Embedding"]
    fd_base = [compute_frechet_distance(stat_dtu[l][0], stat_dtu[l][1], stat_base[l][0], stat_base[l][1]) for l in layers]
    fd_car  = [compute_frechet_distance(stat_dtu[l][0], stat_dtu[l][1], stat_car[l][0], stat_car[l][1]) for l in layers]
    
    plt.figure(figsize=(10, 5))
    plt.plot(layers, fd_base, marker='o', label="KUL Baseline", color='red')
    plt.plot(layers, fd_car, marker='o', label="KUL CAR", color='blue')
    plt.title("Fréchet Distance to DTU across Network Layers")
    plt.ylabel("Fréchet Distance (lower is better)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("layer_trajectory.png")
    plt.close()
    
    # 3. Embedding PCA
    all_embs = np.vstack([emb_dtu, emb_base, emb_car])
    labels = ["DTU"]*len(emb_dtu) + ["Baseline"]*len(emb_base) + ["CAR"]*len(emb_car)
    
    pca = PCA(n_components=2)
    pca_embs = pca.fit_transform(all_embs)
    
    pca_dtu = pca_embs[:len(emb_dtu)]
    pca_base = pca_embs[len(emb_dtu):len(emb_dtu)+len(emb_base)]
    pca_car = pca_embs[len(emb_dtu)+len(emb_base):]
    
    plt.figure(figsize=(8, 8))
    plt.scatter(pca_dtu[:, 0], pca_dtu[:, 1], alpha=0.3, label="DTU (Target)", color='green')
    plt.scatter(pca_base[:, 0], pca_base[:, 1], alpha=0.5, label="KUL Baseline", color='red')
    plt.scatter(pca_car[:, 0], pca_car[:, 1], alpha=0.5, label="KUL CAR", color='blue')
    plt.title("PCA of 64-D Latent Embeddings")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("embedding_pca.png")
    plt.close()
    
    print("\nSuccessfully generated 'layer_trajectory.png' and 'embedding_pca.png'.")

if __name__ == "__main__":
    run_experiment()
