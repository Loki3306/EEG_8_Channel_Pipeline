import os
import sys
import pickle
import numpy as np
import scipy.linalg
from scipy.signal import butter, filtfilt, resample_poly
import scipy.io as sio
import torch
import torch.nn.functional as F
from tqdm import tqdm
import math

# Add repository root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.ridge_aad import load_subject_examples, subject_files
from data.extract_gammatone_envelopes import extract_gammatone_envelopes
from analysis.experiment_8_dtu_fingerprint import get_dtu_mapping_and_envelopes, normalize_array
from models.matchnet import ContrastiveMatchNet

def load_matchnet(device='cuda'):
    model = ContrastiveMatchNet(eeg_model_type="eegnet", eeg_channels=8, audio_channels=28, latent_dim=64)
    chk_path = "/kaggle/working/EEG_8_Channel_Pipeline/checkpoints/matchnet_fold_S2_data_preproc_best.pth"
    if os.path.exists(chk_path):
        model.load_state_dict(torch.load(chk_path, map_location=device))
    model.to(device)
    model.eval()
    return model

class LayerProfiler:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.activations = {}
        
        self.hooks = []
        self._register_hooks()
        
    def _register_hooks(self):
        def get_hook(name):
            def hook(module, input, output):
                # Output shape from EEGNet blocks: [B, C, 1, T] or [B, C, T]
                # Output from Audio blocks: [B, C, T]
                # We want to Global Average Pool over T (and spatial 1 if it exists)
                
                # Check if it has 4 dims [B, C, H, W] (e.g. EEGNet block output)
                if len(output.shape) == 4:
                    pooled = output.mean(dim=(2, 3)) # -> [B, C]
                # Check if it has 3 dims [B, C, T] (e.g. Conv1d or latent)
                elif len(output.shape) == 3:
                    pooled = output.mean(dim=2) # -> [B, C]
                else:
                    pooled = output # Should not happen
                    
                if name not in self.activations:
                    self.activations[name] = []
                self.activations[name].append(pooled.detach().cpu().numpy()[0])
            return hook
            
        # EEG Branch
        self.hooks.append(self.model.eeg_encoder.block1.register_forward_hook(get_hook("EEG_Block1")))
        self.hooks.append(self.model.eeg_encoder.block2.register_forward_hook(get_hook("EEG_Block2")))
        self.hooks.append(self.model.eeg_encoder.output_proj.register_forward_hook(get_hook("EEG_Embedding")))
        
        # Audio Branch
        self.hooks.append(self.model.audio_encoder.net[3].register_forward_hook(get_hook("Audio_Conv1")))
        self.hooks.append(self.model.audio_encoder.net[7].register_forward_hook(get_hook("Audio_Conv2")))
        self.hooks.append(self.model.audio_encoder.net[8].register_forward_hook(get_hook("Audio_Embedding")))
        
    def clear(self):
        self.activations = {k: [] for k in self.activations}
        
    def get_activations(self):
        return {k: np.array(v) for k, v in self.activations.items()}
        
    def remove_hooks(self):
        for h in self.hooks:
            h.remove()

def compute_dtu_activations(profiler, fs_target=64, win_len_sec=3, stride_sec=1.5):
    print("\n--- Extracting DTU Layer Activations ---")
    s_files = subject_files()
    if not s_files:
        print("ERROR: DTU Dataset not found!")
        return None
        
    mapping, envelopes = get_dtu_mapping_and_envelopes()
    dtu_indices = [13, 46, 43, 23, 50, 0, 52, 14] # The exact training channels
    
    profiler.clear()
    
    for s_file in tqdm(s_files, desc="DTU Subjects"):
        subj = s_file.stem.split('_')[0]
        examples = load_subject_examples(s_file)
        
        for ex in examples:
            eeg_data = ex.eeg_data
            eeg_8 = eeg_data[:, dtu_indices]
            eeg_norm = normalize_array(eeg_8)
            
            trial_key = f"trial{ex.trial_index+1}"
            if subj in mapping and trial_key in mapping[subj]:
                fname_a = mapping[subj][trial_key]["wavA"]["filename"]
                fname_b = mapping[subj][trial_key]["wavB"]["filename"]
                
                if fname_a in envelopes and fname_b in envelopes:
                    env_a = envelopes[fname_a]
                    env_b = envelopes[fname_b]
                    
                    env_att = env_a if ex.label == 1 else env_b
                    env_unatt = env_b if ex.label == 1 else env_a
                    
                    env_att = normalize_array(env_att.T).T
                    env_unatt = normalize_array(env_unatt.T).T
                    
                    min_len = min(len(eeg_norm), env_att.shape[1], env_unatt.shape[1])
                    win_len = int(win_len_sec * fs_target)
                    stride = int(stride_sec * fs_target)
                    
                    for start in range(0, min_len - win_len + 1, stride):
                        eeg_win = eeg_norm[start:start+win_len]
                        att_win = env_att[:, start:start+win_len]
                        unatt_win = env_unatt[:, start:start+win_len]
                        
                        e_tensor = torch.tensor(eeg_win.T, dtype=torch.float32).unsqueeze(0).to(profiler.device)
                        a_tensor = torch.tensor(att_win, dtype=torch.float32).unsqueeze(0).to(profiler.device)
                        u_tensor = torch.tensor(unatt_win, dtype=torch.float32).unsqueeze(0).to(profiler.device)
                        
                        with torch.no_grad():
                            profiler.model(e_tensor, a_tensor, u_tensor)
                            
    return profiler.get_activations()

def compute_kul_activations(profiler, fs_target=64, win_len_sec=3, stride_sec=1.5):
    print("\n--- Extracting KUL Layer Activations ---")
    
    target_channels = ['T7', 'C2', 'FT8', 'P7', 'CPz', 'Fp1', 'TP8', 'C3']
    mat_path = "/kaggle/input/datasets/lowk1ee/s1-klu/S1_KLU.mat"
    wav_dir = "/kaggle/input/datasets/lowk1ee/audio-klu"
    
    if not os.path.exists(mat_path):
        print("ERROR: KUL Dataset not found!")
        return None
        
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    trials = mat['trials'] if 'trials' in mat else mat['trial']
    
    profiler.clear()
    
    def find_wav(name):
        for r, d, f in os.walk(wav_dir):
            if name in f: return os.path.join(r, name)
            if name+".wav" in f: return os.path.join(r, name+".wav")
        return None
        
    audio_cache = {}
    
    for trial in tqdm(trials, desc="KUL Trials"):
        eeg_data = trial.RawData.EegData
        fs_eeg = trial.FileHeader.SampleRate
        channel_names = [ch.Label for ch in trial.FileHeader.Channels]
        
        try:
            sel_idx = [channel_names.index(tc) if tc in channel_names else [c.upper() for c in channel_names].index(tc.upper()) for tc in target_channels]
        except ValueError:
            continue
            
        eeg_8 = eeg_data[:, sel_idx]
        
        nyq = 0.5 * fs_eeg
        b, a = butter(2, [1.0/nyq, 8.0/nyq], btype='band')
        eeg_8 = filtfilt(b, a, eeg_8, axis=0)
        
        g = math.gcd(fs_target, fs_eeg)
        eeg_8 = resample_poly(eeg_8, fs_target // g, fs_eeg // g, axis=0)
        eeg_norm = normalize_array(eeg_8)
        
        att_ear = trial.attended_ear
        stimuli = trial.stimuli
        att_wav_name = str(stimuli[0] if att_ear == 'L' else stimuli[1])
        unatt_wav_name = str(stimuli[1] if att_ear == 'L' else stimuli[0])
        
        att_wav_path = find_wav(att_wav_name)
        unatt_wav_path = find_wav(unatt_wav_name)
        
        if att_wav_path and unatt_wav_path:
            if att_wav_name not in audio_cache:
                audio_cache[att_wav_name] = extract_gammatone_envelopes(att_wav_path, target_fs=fs_target)
            if unatt_wav_name not in audio_cache:
                audio_cache[unatt_wav_name] = extract_gammatone_envelopes(unatt_wav_path, target_fs=fs_target)
                
            env_att = audio_cache[att_wav_name]
            env_unatt = audio_cache[unatt_wav_name]
            
            env_att = normalize_array(env_att.T).T
            env_unatt = normalize_array(env_unatt.T).T
            
            min_len = min(len(eeg_norm), env_att.shape[1], env_unatt.shape[1])
            win_len = int(win_len_sec * fs_target)
            stride = int(stride_sec * fs_target)
            
            for start in range(0, min_len - win_len + 1, stride):
                eeg_win = eeg_norm[start:start+win_len]
                att_win = env_att[:, start:start+win_len]
                unatt_win = env_unatt[:, start:start+win_len]
                
                e_tensor = torch.tensor(eeg_win.T, dtype=torch.float32).unsqueeze(0).to(profiler.device)
                a_tensor = torch.tensor(att_win, dtype=torch.float32).unsqueeze(0).to(profiler.device)
                u_tensor = torch.tensor(unatt_win, dtype=torch.float32).unsqueeze(0).to(profiler.device)
                
                with torch.no_grad():
                    profiler.model(e_tensor, a_tensor, u_tensor)
                    
    return profiler.get_activations()

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Computes the Fréchet distance (Wasserstein-2) between two multivariate Gaussians.
    d^2 = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
    """
    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def run_layer_profiling():
    print("="*70)
    print("PHASE C: MATCHNET LAYER-WISE STATISTICAL PROFILER")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_matchnet(device)
    if model is None:
        return
        
    profiler = LayerProfiler(model, device)
    
    # Check if we already cached DTU to save time
    os.makedirs('data', exist_ok=True)
    dtu_cache_path = 'data/DTU_Layer_Activations.pkl'
    if os.path.exists(dtu_cache_path):
        print(f"Loading DTU activations from cache {dtu_cache_path}...")
        with open(dtu_cache_path, 'rb') as f:
            dtu_act = pickle.load(f)
    else:
        dtu_act = compute_dtu_activations(profiler)
        if dtu_act:
            with open(dtu_cache_path, 'wb') as f:
                pickle.dump(dtu_act, f)
                
    kul_act = compute_kul_activations(profiler)
    
    if not dtu_act or not kul_act:
        print("Missing dataset. Aborting.")
        return
        
    print("\n" + "="*70)
    print("DISTRIBUTION DIVERGENCE (Fréchet Distance & Cosine Distance)")
    print("="*70)
    
    layers = [
        "EEG_Block1", "EEG_Block2", "EEG_Embedding",
        "Audio_Conv1", "Audio_Conv2", "Audio_Embedding"
    ]
    
    print(f"{'Layer':<20} | {'Dim':<5} | {'Cosine Dist':<15} | {'Fréchet Distance'}")
    print("-" * 70)
    
    for layer in layers:
        if layer not in dtu_act or layer not in kul_act: continue
        
        # dtu_arr and kul_arr shape: [N_windows, Channels]
        dtu_arr = dtu_act[layer]
        kul_arr = kul_act[layer]
        
        if len(dtu_arr) == 0 or len(kul_arr) == 0: continue
        
        # Means
        mu_d = np.mean(dtu_arr, axis=0)
        mu_k = np.mean(kul_arr, axis=0)
        
        # Cosine distance of centroids
        cos_dist = 1 - np.dot(mu_d, mu_k) / (np.linalg.norm(mu_d) * np.linalg.norm(mu_k) + 1e-12)
        
        # Covariances
        sig_d = np.cov(dtu_arr, rowvar=False)
        sig_k = np.cov(kul_arr, rowvar=False)
        
        if sig_d.ndim == 0:
            sig_d = np.array([[sig_d]])
            sig_k = np.array([[sig_k]])
            
        fd = calculate_frechet_distance(mu_d, sig_d, mu_k, sig_k)
        
        print(f"{layer:<20} | {dtu_arr.shape[1]:<5} | {cos_dist:<15.4f} | {fd:.4f}")

if __name__ == "__main__":
    run_layer_profiling()
