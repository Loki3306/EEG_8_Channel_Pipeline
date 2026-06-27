import os
import sys
import math
import numpy as np
import scipy.io
import scipy.signal
import scipy.linalg
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.experiment_16_input_equivalence import get_dtu_tensor
from analysis.experiment_17_spatial_filters import compute_band_power, compute_psd
from models.matchnet import ContrastiveMatchNet

def get_kul_tensor(apply_car=False, zero_fp1=False):
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
    
    for trial in trials:
        eeg_data = trial.RawData.EegData
        fs_eeg = trial.FileHeader.SampleRate
        channel_names = [ch.Label for ch in trial.FileHeader.Channels]
        
        target_channels = ['T7', 'C2', 'FT8', 'P7', 'CPz', 'Fp1', 'TP8', 'C3']
        
        if apply_car:
            eeg_data = eeg_data - eeg_data.mean(axis=1, keepdims=True)
            
        try:
            sel_idx = [channel_names.index(tc) if tc in channel_names else [c.upper() for c in channel_names].index(tc.upper()) for tc in target_channels]
        except ValueError as e:
            raise RuntimeError(f"Missing EEG channel in KUL dataset! {e}")
            
        eeg_8 = eeg_data[:, sel_idx]
        
        nyq = 0.5 * fs_eeg
        b, a = scipy.signal.butter(4, [1.0/nyq, 8.0/nyq], btype='band')
        eeg_8 = scipy.signal.filtfilt(b, a, eeg_8, axis=0)
        
        g = math.gcd(64, int(fs_eeg))
        eeg_8 = scipy.signal.resample_poly(eeg_8, 64 // g, int(fs_eeg) // g, axis=0)
        
        if zero_fp1:
            eeg_8[:, 5] = 0.0 # Fp1 is at index 5
            
        # Normalize
        arr = eeg_8 - eeg_8.mean(axis=0, keepdims=True)
        scale = arr.std(axis=0, keepdims=True) + 1e-12
        eeg_norm = arr / scale
        
        att_ear = trial.attended_ear
        stimuli = trial.stimuli
        if len(stimuli) < 2: continue
        att_wav_name = str(stimuli[0] if att_ear == 'L' else stimuli[1])
        unatt_wav_name = str(stimuli[1] if att_ear == 'L' else stimuli[0])
        
        def find_wav(name):
            wav_dir = "/kaggle/input/datasets/lowk1ee/audio-klu" if os.path.exists("/kaggle/input/datasets/lowk1ee/audio-klu") else "data"
            for r, d, f in os.walk(wav_dir):
                for file in f:
                    if name in file and file.endswith(".wav"):
                        return os.path.join(r, file)
            return None
            
        att_wav_path = find_wav(att_wav_name)
        unatt_wav_path = find_wav(unatt_wav_name)
        
        if att_wav_path and unatt_wav_path:
            import torchaudio
            try:
                from data.extract_gammatone_envelopes import extract_gammatone_envelopes
                env_att = extract_gammatone_envelopes(att_wav_path, fs_target=64)
                env_unatt = extract_gammatone_envelopes(unatt_wav_path, fs_target=64)
                
                def norm_env(env):
                    env = env.T
                    env = env - env.mean(axis=0, keepdims=True)
                    env = env / (env.std(axis=0, keepdims=True) + 1e-12)
                    return env.T
                    
                env_att = norm_env(env_att)
                env_unatt = norm_env(env_unatt)
                
                min_len = min(len(eeg_norm), env_att.shape[1], env_unatt.shape[1])
                for start in range(0, min_len - win_len + 1, win_len):
                    e_list.append(eeg_norm[start:start+win_len].T)
                    a_list.append(env_att[:, start:start+win_len])
                    u_list.append(env_unatt[:, start:start+win_len])
            except Exception as e:
                pass
                
    return np.array(e_list)[:100], np.array(a_list)[:100], np.array(u_list)[:100]

class FullLayerProfiler:
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.hooks = []
        self._register_hooks()
        
    def _register_hooks(self):
        def get_hook(name):
            def hook(module, input, output):
                if len(output.shape) == 4:
                    pooled = output.mean(dim=(2, 3))
                elif len(output.shape) == 3:
                    pooled = output.mean(dim=2)
                else:
                    pooled = output
                if name not in self.activations:
                    self.activations[name] = []
                self.activations[name].append(pooled.detach().cpu().numpy()[0])
            return hook
            
        eeg_net = self.model.eeg_encoder
        self.hooks.append(eeg_net.block1[0:2].register_forward_hook(get_hook("TemporalConv")))
        self.hooks.append(eeg_net.block1[2].register_forward_hook(get_hook("SpatialConv")))
        self.hooks.append(eeg_net.block1[3].register_forward_hook(get_hook("BN2")))
        self.hooks.append(eeg_net.block1.register_forward_hook(get_hook("Block1_Out")))
        self.hooks.append(eeg_net.block2.register_forward_hook(get_hook("Block2_Out")))
        self.hooks.append(eeg_net.output_proj.register_forward_hook(get_hook("Embedding")))
        
    def clear(self):
        self.activations = {k: [] for k in self.activations}
        
    def get_activations(self):
        return {k: np.array(v) for k, v in self.activations.items()}

def compute_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)

def evaluate_condition(model, profiler, e_tensor, a_tensor, u_tensor):
    profiler.clear()
    margins = []
    correct = 0
    
    # We will process one by one to use the profiler exactly like training batches
    for i in range(len(e_tensor)):
        e = torch.tensor(e_tensor[i], dtype=torch.float32).unsqueeze(0).unsqueeze(1)
        a = torch.tensor(a_tensor[i], dtype=torch.float32).unsqueeze(0)
        u = torch.tensor(u_tensor[i], dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            emb_e, emb_a, emb_u = model(e, a, u)
            sim_a = torch.nn.functional.cosine_similarity(emb_e, emb_a).item()
            sim_u = torch.nn.functional.cosine_similarity(emb_e, emb_u).item()
            
            margin = sim_a - sim_u
            margins.append(margin)
            if margin > 0:
                correct += 1
                
    acc = correct / len(e_tensor)
    acts = profiler.get_activations()
    
    # Compute Gaussian stats for Fréchet Distance
    stats = {}
    for layer, A in acts.items():
        mu = np.mean(A, axis=0)
        sigma = np.cov(A, rowvar=False)
        stats[layer] = (mu, sigma)
        
    # Spatial Correlation
    # e_tensor shape: (N, C, T) -> flatten to (C, N*T)
    C = e_tensor.shape[1]
    e_flat = e_tensor.transpose(1, 0, 2).reshape(C, -1)
    spatial_corr = np.corrcoef(e_flat)
    
    # Band Powers
    freqs, psds = compute_psd(e_tensor, fs=64)
    band_powers = compute_band_power(freqs, psds)
    avg_band_powers = {k: np.mean(v) for k, v in band_powers.items()}
    
    return acc, margins, stats, spatial_corr, avg_band_powers

def print_histogram(margins, bins=20):
    margins = np.array(margins)
    counts, edges = np.histogram(margins, bins=bins, range=(-1.0, 1.0))
    max_count = max(counts) if len(counts) > 0 and max(counts) > 0 else 1
    
    print("\nMargin Distribution (simA - simU):")
    print("<-1.0 (Unattended)                              0.0                              (Attended) >1.0")
    print("-" * 96)
    
    # Scale counts to max 20 blocks
    for i in range(len(counts)):
        bar_len = int(20 * counts[i] / max_count)
        bar = "█" * bar_len
        center = (edges[i] + edges[i+1]) / 2
        print(f"{center:>5.2f} | {bar}")

def run_experiment():
    model_path = None
    for r, d, f in os.walk("/kaggle/input"):
        for file in f:
            if file in ["best_model.pt", "best_model.pth", "checkpoint.pt", "checkpoint.pth", "matchnet.pt", "matchnet.pth"]:
                model_path = os.path.join(r, file)
                break
        if model_path: break
        
    if not model_path:
        for r, d, f in os.walk("checkpoints"):
            for file in f:
                if file in ["best_model.pt", "best_model.pth", "checkpoint.pt", "checkpoint.pth", "matchnet.pt", "matchnet.pth"]:
                    model_path = os.path.join(r, file)
                    break
            if model_path: break
            
    if not model_path:
        print("Could not find MatchNet checkpoint. Cannot evaluate accuracy and layers.")
        return

    model = ContrastiveMatchNet(eeg_model_type="eegnet", eeg_channels=8, audio_channels=28, latent_dim=64)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    profiler = FullLayerProfiler(model)
    
    print("Loading DTU Target Representations...")
    e_dtu, a_dtu, u_dtu = get_dtu_tensor()
    if e_dtu is None or len(e_dtu) == 0:
        print("Failed to load DTU tensors.")
        return
        
    acc_dtu, mar_dtu, stat_dtu, corr_dtu, bp_dtu = evaluate_condition(model, profiler, e_dtu, a_dtu, u_dtu)
    
    conditions = [
        ("Condition A: Current KUL (Baseline)", False, False),
        ("Condition B: KUL + CAR Recovery", True, False),
        ("Condition C: KUL + CAR + No Fp1", True, True)
    ]
    
    for name, apply_car, zero_fp1 in conditions:
        print(f"\n{'='*80}\n{name}\n{'='*80}")
        e_kul, a_kul, u_kul = get_kul_tensor(apply_car, zero_fp1)
        if e_kul is None or len(e_kul) == 0:
            print("Failed to load KUL tensors.")
            continue
            
        acc_kul, mar_kul, stat_kul, corr_kul, bp_kul = evaluate_condition(model, profiler, e_kul, a_kul, u_kul)
        
        print(f"Accuracy: {acc_kul*100:.2f}% (DTU Target: {acc_dtu*100:.2f}%)")
        print(f"Mean Margin: {np.mean(mar_kul):.4f} (DTU Target: {np.mean(mar_dtu):.4f})")
        print_histogram(mar_kul)
        
        print("\nLayer Fréchet Distances (vs DTU):")
        print(f"{'Layer':<15} | {'FD':<10}")
        print("-" * 30)
        
        for layer in stat_dtu.keys():
            mu1, sig1 = stat_dtu[layer]
            mu2, sig2 = stat_kul[layer]
            fd = compute_frechet_distance(mu1, sig1, mu2, sig2)
            print(f"{layer:<15} | {fd:<10.2f}")
            
        print("\nBand Power (KUL / DTU Ratio):")
        for b in bp_dtu.keys():
            ratio = bp_kul[b] / bp_dtu[b] if bp_dtu[b] > 0 else 0
            print(f"{b:<15}: {ratio:.2f}")

if __name__ == "__main__":
    run_experiment()
