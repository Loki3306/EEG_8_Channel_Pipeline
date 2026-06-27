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
from models.matchnet import ContrastiveMatchNet

def get_kul_tensor(apply_car=False, zero_fp1=False):
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
    
    for t_idx, trial in enumerate(trials):
        eeg_data = trial.RawData.EegData
        fs_eeg = trial.FileHeader.SampleRate
        channel_names = [ch.Label for ch in trial.FileHeader.Channels]
        
        target_channels = ['T7', 'C2', 'FT8', 'P7', 'CPz', 'Fp1', 'TP8', 'C3']
        
        if apply_car:
            # Recover CAR by subtracting the mean of all 64 available channels at each timepoint
            eeg_data = eeg_data - eeg_data.mean(axis=0, keepdims=True)
            
        try:
            sel_idx = [channel_names.index(tc) if tc in channel_names else [c.upper() for c in channel_names].index(tc.upper()) for tc in target_channels]
        except ValueError as e:
            raise RuntimeError(f"Missing EEG channel in KUL dataset! {e}")
            
        eeg_8 = eeg_data[sel_idx, :].T # shape: (T, 8)
        
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
                    if file == name or file == name + ".wav":
                        return os.path.join(r, file)
            return None
            
        att_wav_path = find_wav(att_wav_name)
        unatt_wav_path = find_wav(unatt_wav_name)
        
        if att_wav_path and unatt_wav_path:
            import torchaudio
            try:
                from data.extract_gammatone_envelopes import extract_gammatone_envelopes
                env_att = extract_gammatone_envelopes(att_wav_path, target_fs=64)
                env_unatt = extract_gammatone_envelopes(unatt_wav_path, target_fs=64)
                
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
                    t_idx_list.append(t_idx)
            except Exception as e:
                print(f"Error extracting audio: {e}")
                
    return np.array(e_list)[:100], np.array(a_list)[:100], np.array(u_list)[:100], np.array(t_idx_list)[:100]

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
    sigma1 += np.eye(sigma1.shape[0]) * eps
    sigma2 += np.eye(sigma2.shape[0]) * eps
    
    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)

def compute_cosine_similarity(mu1, mu2):
    return np.dot(mu1, mu2) / (np.linalg.norm(mu1) * np.linalg.norm(mu2) + 1e-12)

def evaluate_condition(model, profiler, e_tensor, a_tensor, u_tensor, t_idx_list=None):
    profiler.clear()
    margins = []
    
    # Store margins per trial
    trial_margins = {}
    
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
            
            if t_idx_list is not None:
                t_idx = t_idx_list[i]
                if t_idx not in trial_margins:
                    trial_margins[t_idx] = []
                trial_margins[t_idx].append(margin)
                
    win_acc = sum(1 for m in margins if m > 0) / len(margins)
    
    trial_acc = 0.0
    if trial_margins:
        correct_trials = 0
        for t_idx, t_margins in trial_margins.items():
            if np.mean(t_margins) > 0:
                correct_trials += 1
        trial_acc = correct_trials / len(trial_margins)
    
    acts = profiler.get_activations()
    
    stats = {}
    for layer, A in acts.items():
        mu = np.mean(A, axis=0)
        sigma = np.cov(A, rowvar=False)
        stats[layer] = (mu, sigma)
        
    return win_acc, trial_acc, margins, stats

def print_histogram(margins, bins=20):
    margins = np.array(margins)
    counts, edges = np.histogram(margins, bins=bins, range=(-1.0, 1.0))
    max_count = max(counts) if len(counts) > 0 and max(counts) > 0 else 1
    
    print("\nMargin Distribution (simA - simU):")
    print("<-1.0 (Unattended)                              0.0                              (Attended) >1.0")
    print("-" * 96)
    
    for i in range(len(counts)):
        bar_len = int(20 * counts[i] / max_count)
        bar = "█" * bar_len
        center = (edges[i] + edges[i+1]) / 2
        print(f"{center:>5.2f} | {bar}")

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
        
    acc_dtu, _, mar_dtu, stat_dtu = evaluate_condition(model, profiler, e_dtu, a_dtu, u_dtu)
    
    conditions = [
        ("Baseline", False, False),
        ("CAR", True, False),
        ("CAR+Fp1Zero", True, True)
    ]
    
    results = {}
    
    for name, apply_car, zero_fp1 in conditions:
        print(f"\n{'='*80}\nCondition: {name}\n{'='*80}")
        e_kul, a_kul, u_kul, t_idx_kul = get_kul_tensor(apply_car, zero_fp1)
        if e_kul is None or len(e_kul) == 0:
            print("Failed to load KUL tensors.")
            continue
            
        win_acc, trial_acc, mar_kul, stat_kul = evaluate_condition(model, profiler, e_kul, a_kul, u_kul, t_idx_kul)
        
        print(f"Window Accuracy: {win_acc*100:.2f}%")
        print(f"Trial Accuracy:  {trial_acc*100:.2f}%")
        print(f"Mean Margin:     {np.mean(mar_kul):.4f}")
        print_histogram(mar_kul)
        
        print("\nLayer Representations (vs DTU):")
        print(f"{'Layer':<15} | {'Fréchet Dist':<12} | {'Cosine(mu)':<12}")
        print("-" * 45)
        
        layer_metrics = {}
        for layer in stat_dtu.keys():
            mu1, sig1 = stat_dtu[layer]
            mu2, sig2 = stat_kul[layer]
            fd = compute_frechet_distance(mu1, sig1, mu2, sig2)
            cos = compute_cosine_similarity(mu1, mu2)
            layer_metrics[layer] = {'fd': fd, 'cos': cos}
            print(f"{layer:<15} | {fd:<12.2f} | {cos:<12.4f}")
            
        results[name] = {
            'win_acc': win_acc,
            'trial_acc': trial_acc,
            'margin': np.mean(mar_kul),
            'block1_fd': layer_metrics.get('Block1_Out', {}).get('fd', 0),
            'emb_fd': layer_metrics.get('Embedding', {}).get('fd', 0)
        }
        
    print("\n" + "="*80)
    print("FINAL SUMMARY TABLE")
    print("="*80)
    print(f"| {'Condition':<12} | {'Trial Acc':<10} | {'Window Acc':<10} | {'Margin':<8} | {'Block1 FD':<10} | {'Embed FD':<10} |")
    print(f"|{'-'*14}|{'-'*12}|{'-'*12}|{'-'*10}|{'-'*12}|{'-'*12}|")
    for name in [c[0] for c in conditions]:
        if name in results:
            r = results[name]
            print(f"| {name:<12} | {r['trial_acc']*100:>8.1f}% | {r['win_acc']*100:>8.1f}% | {r['margin']:>8.4f} | {r['block1_fd']:>9.2f} | {r['emb_fd']:>9.2f} |")

if __name__ == "__main__":
    run_experiment()
