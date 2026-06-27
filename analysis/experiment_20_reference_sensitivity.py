import os
import sys
import math
import numpy as np
import scipy.io
import scipy.signal
import scipy.linalg
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.experiment_16_input_equivalence import get_dtu_tensor
from analysis.experiment_18_car_ablation import FullLayerProfiler, compute_frechet_distance
from models.matchnet import ContrastiveMatchNet

def get_kul_tensors_custom(alpha=0.0, zero_channels=None, windows_per_trial=20):
    """
    alpha: 0.0 means Cz reference, 1.0 means CAR. 
           Mathematically: EEG_alpha = EEG_cz - alpha * mean(EEG_cz, axis=1)
    zero_channels: list of target channel indices to zero out.
    """
    if zero_channels is None: zero_channels = []
    
    mat_path = "/kaggle/input/datasets/lowk1ee/s1-klu/S1_KLU.mat"
    if not os.path.exists(mat_path):
        mat_path = "data/S1_KLU.mat"
    if not os.path.exists(mat_path):
        print("Missing KUL data.")
        return None, None, None
        
    mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    trials = mat['trials'] if 'trials' in mat else mat['trial']
    e_list, a_list, u_list = [], [], []
    win_len = int(3 * 64)
    
    cache_path = "kul_gammatone_cache.pkl"
    if os.path.exists(cache_path):
        import pickle
        with open(cache_path, "rb") as f:
            envelope_cache = pickle.load(f)
    else:
        envelope_cache = {}
    
    target_channels = ['T7', 'C2', 'FT8', 'P7', 'CPz', 'Fp1', 'TP8', 'C3']
    
    for t_idx, trial in enumerate(trials):
        eeg_data = trial.RawData.EegData
        fs_eeg = trial.FileHeader.SampleRate
        channel_names = [ch.Label for ch in trial.FileHeader.Channels]
        
        # Apply alpha interpolation
        if alpha != 0.0:
            eeg_data = eeg_data - alpha * eeg_data.mean(axis=1, keepdims=True)
            
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
        
        for zc in zero_channels:
            eeg_8[:, zc] = 0.0
            
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
                    
    return np.array(e_list), np.array(a_list), np.array(u_list)

def evaluate_fast(model, profiler, e_tensor, a_tensor, u_tensor):
    profiler.clear()
    margins = []
    preds = []
    
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
            preds.append(1 if margin > 0 else 0)
                
    acts = profiler.get_activations()
    stats = {}
    for layer, A in acts.items():
        mu = np.mean(A, axis=0)
        sigma = np.cov(A, rowvar=False)
        stats[layer] = (mu, sigma)
        
    return margins, preds, stats

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
    _, _, stat_dtu = evaluate_fast(model, profiler, e_dtu, a_dtu, u_dtu)
    
    print("\n" + "="*80)
    print("1. REFERENCE INTERPOLATION (0% - 100% CAR)")
    print("="*80)
    alphas = np.linspace(0.0, 1.0, 11)
    accs, mean_margins, fds = [], [], []
    
    for alpha in alphas:
        e_k, a_k, u_k = get_kul_tensors_custom(alpha=alpha)
        m, p, s = evaluate_fast(model, profiler, e_k, a_k, u_k)
        acc = np.mean(p)
        mar = np.mean(m)
        fd = compute_frechet_distance(stat_dtu["Block2_Out"][0], stat_dtu["Block2_Out"][1], s["Block2_Out"][0], s["Block2_Out"][1])
        accs.append(acc)
        mean_margins.append(mar)
        fds.append(fd)
        print(f"Alpha {alpha*100:>5.1f}% | Acc: {acc*100:>5.1f}% | Mar: {mar:>7.4f} | Block2 FD: {fd:>7.2f}")
        
    # Plot Interpolation
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    ax1.plot(alphas, accs, 'g-o', label='Accuracy')
    ax2.plot(alphas, fds, 'r-s', label='Block2 FD')
    ax1.set_xlabel('CAR Interpolation (\u03B1)')
    ax1.set_ylabel('Accuracy', color='g')
    ax2.set_ylabel('Block2 Fréchet Distance', color='r')
    plt.title('Performance vs CAR Interpolation')
    fig.tight_layout()
    plt.savefig('reference_interpolation.png')
    plt.close()

    print("\n" + "="*80)
    print("2. CHANNEL IMPORTANCE ABLATION (on 100% CAR)")
    print("="*80)
    channels = ['T7', 'C2', 'FT8', 'P7', 'CPz', 'Fp1', 'TP8', 'C3']
    
    # Get 100% CAR baseline
    e_car, a_car, u_car = get_kul_tensors_custom(alpha=1.0)
    m_car, p_car, s_car = evaluate_fast(model, profiler, e_car, a_car, u_car)
    base_acc = np.mean(p_car)
    base_mar = np.mean(m_car)
    
    print(f"CAR Baseline    | Acc: {base_acc*100:>5.1f}% | Mar: {base_mar:>7.4f}")
    print("-" * 65)
    
    for i, ch in enumerate(channels):
        e_z, a_z, u_z = get_kul_tensors_custom(alpha=1.0, zero_channels=[i])
        m_z, p_z, s_z = evaluate_fast(model, profiler, e_z, a_z, u_z)
        z_acc = np.mean(p_z)
        z_mar = np.mean(m_z)
        delta_acc = z_acc - base_acc
        delta_mar = z_mar - base_mar
        print(f"Zero {ch:<10} | Acc: {z_acc*100:>5.1f}% (\u0394 {delta_acc*100:>5.1f}%) | Mar: {z_mar:>7.4f} (\u0394 {delta_mar:>7.4f})")

    print("\n" + "="*80)
    print("3. BLOCK 1 FILTER VISUALIZATION")
    print("="*80)
    spatial_weights = model.eeg_encoder.block1[2].weight.detach().cpu().numpy() # Shape: (16, 1, 8, 1)
    spatial_weights = spatial_weights.squeeze() # Shape: (16, 8)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(spatial_weights, aspect='auto', cmap='coolwarm', vmin=-np.max(np.abs(spatial_weights)), vmax=np.max(np.abs(spatial_weights)))
    plt.colorbar(label='Weight')
    plt.xticks(ticks=np.arange(8), labels=channels)
    plt.yticks(ticks=np.arange(16), labels=[f"Filter {i}" for i in range(16)])
    plt.title('MatchNet Spatial Filters (SpatialConv)')
    plt.tight_layout()
    plt.savefig('spatial_filters.png')
    plt.close()
    
    print("Saved 'spatial_filters.png'")
    
    print("\n" + "="*80)
    print("4. WINDOW-LEVEL RESCUE ANALYSIS")
    print("="*80)
    # Get 0% Baseline
    e_base, a_base, u_base = get_kul_tensors_custom(alpha=0.0)
    m_base, p_base, _ = evaluate_fast(model, profiler, e_base, a_base, u_base)
    
    ww, wc, cc, cw = 0, 0, 0, 0
    rescued_margins = []
    
    for pb, pc, mb in zip(p_base, p_car, m_base):
        if pb == 0 and pc == 0: ww += 1
        elif pb == 0 and pc == 1: 
            wc += 1
            rescued_margins.append(mb)
        elif pb == 1 and pc == 1: cc += 1
        elif pb == 1 and pc == 0: cw += 1
        
    print(f"Total Windows Evaluated: {len(p_base)}")
    print(f"Wrong -> Wrong:   {ww}")
    print(f"Wrong -> Correct: {wc}  (Rescued by CAR)")
    print(f"Correct -> Correct: {cc}")
    print(f"Correct -> Wrong:   {cw}")
    
    if len(rescued_margins) > 0:
        print(f"\nStats for Rescued Windows:")
        print(f"Average original margin before rescue: {np.mean(rescued_margins):.4f}")
        
    print("\nSuccessfully generated 'reference_interpolation.png' and 'spatial_filters.png'.")

if __name__ == "__main__":
    run_experiment()
