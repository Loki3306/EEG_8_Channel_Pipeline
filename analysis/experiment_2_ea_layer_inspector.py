import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.io.wavfile as wavfile
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.signal import resample, butter, filtfilt
from sklearn.decomposition import PCA
from tqdm import tqdm

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from models.matchnet import ContrastiveMatchNet
from baselines.ridge_aad import load_subject_examples
from preprocessing.euclidean_alignment import prepare_alignment_matrices, apply_alignment

def normalize_array(arr):
    return (arr - arr.mean(axis=0, keepdims=True)) / (arr.std(axis=0, keepdims=True) + 1e-12)

# --- KUL Audio extraction utilities ---
def erb_space(low_freq, high_freq, num_bands):
    erb_low = 21.4 * np.log10(4.37 * low_freq / 1000 + 1)
    erb_high = 21.4 * np.log10(4.37 * high_freq / 1000 + 1)
    erb_points = np.linspace(erb_low, erb_high, num_bands)
    cf = (10 ** (erb_points / 21.4) - 1) * 1000 / 4.37
    return cf

def get_erb_bands(cfs):
    bws = 24.7 * (4.37 * cfs / 1000 + 1)
    return cfs - bws / 2, cfs + bws / 2

def apply_bandpass(data, low, high, fs, order=2):
    nyq = 0.5 * fs
    low = max(low / nyq, 0.001)
    high = min(high / nyq, 0.999)
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def _process_band(audio_data, low, high, fs_in):
    band_audio = apply_bandpass(audio_data, low, high, fs_in)
    return np.abs(band_audio) ** 0.3

def extract_28_band_envelope(audio_data, fs_in, fs_out=64, num_bands=28):
    cfs = erb_space(50, 8000, num_bands)
    lows, highs = get_erb_bands(cfs)
    
    from joblib import Parallel, delayed
    envelopes = Parallel(n_jobs=-1)(
        delayed(_process_band)(audio_data, lows[i], highs[i], fs_in) 
        for i in range(num_bands)
    )
    
    envelopes = np.array(envelopes)
    num_samples_out = int(envelopes.shape[1] * fs_out / fs_in)
    return resample(envelopes, num_samples_out, axis=1)

class EALayerInspector:
    def __init__(self, chk_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ContrastiveMatchNet(eeg_model_type="eegnet", eeg_channels=8, audio_channels=28, latent_dim=64).to(self.device)
        self.model.load_state_dict(torch.load(chk_path, map_location=self.device))
        self.datasets = ["DTU", "KUL", "KUL_EA", "KUL_EA_DTU"]
        self.activations = {d: {} for d in self.datasets}
        self.current_dataset = "DTU"
        
        self._register_hooks()
        
        # We store input data to pass through later
        self.dtu_eeg, self.dtu_aa, self.dtu_ab = [], [], []
        
        self.kul_base_eeg, self.kul_ea_eeg, self.kul_ea_dtu_eeg = [], [], []
        self.kul_aa, self.kul_ab = [], []

    def _register_hooks(self):
        def get_hook(name):
            def hook(model, input, output):
                if name not in self.activations[self.current_dataset]:
                    self.activations[self.current_dataset][name] = []
                # Detach and move to CPU to save memory
                self.activations[self.current_dataset][name].append(output.detach().cpu())
            return hook
            
        # EEG Branch
        self.model.eeg_encoder.block1.register_forward_hook(get_hook("EEG_Block1"))
        self.model.eeg_encoder.block2.register_forward_hook(get_hook("EEG_Block2"))
        self.model.eeg_encoder.output_proj.register_forward_hook(get_hook("EEG_Embedding"))
        
        # Audio Branch
        self.model.audio_encoder.net[3].register_forward_hook(get_hook("Audio_Conv1"))
        self.model.audio_encoder.net[7].register_forward_hook(get_hook("Audio_Conv2"))
        self.model.audio_encoder.net[8].register_forward_hook(get_hook("Audio_Embedding"))

    def load_data(self):
        print("--- LOADING SMALL BATCH OF DATA ---")
        self._load_dtu()
        self._load_kul()
        
    def _load_dtu(self):
        print("Loading DTU Data (S1, First Trial)...")
        map_file = REPO_ROOT / "data" / "audio_mapping.json"
        kaggle_dir = Path("/kaggle/input/datasets/lokeshgile/gammatone-envelope")
        env_file = list(kaggle_dir.glob("*.pkl"))[0] if kaggle_dir.exists() and list(kaggle_dir.glob("*.pkl")) else REPO_ROOT / "data" / "gammatone_envelopes.pkl"
        
        with open(map_file, 'r') as f: mapping = json.load(f)
        with open(env_file, 'rb') as f: envelopes = pickle.load(f)
        
        kaggle_path = Path("/kaggle/input/datasets/lokeshgile/dataset-eeg/S1_data_preproc.mat")
        sub_path = kaggle_path if kaggle_path.exists() else REPO_ROOT / "data" / "S1_data_preproc.mat"
        examples = load_subject_examples(sub_path)
        
        ex = examples[0] # Just need one trial
        dtu_channels = [13, 46, 43, 23, 50, 0, 52, 14]
        eeg = ex.eeg[:, dtu_channels] if ex.eeg.shape[1] > 8 else ex.eeg
        eeg = normalize_array(eeg)
        
        trial_key = f"trial_{ex.trial_index}"
        fname_a = mapping["S1"][trial_key]["wavA"]["filename"]
        fname_b = mapping["S1"][trial_key]["wavB"]["filename"]
        env_a = normalize_array(envelopes[fname_a].T)
        env_b = normalize_array(envelopes[fname_b].T)
        
        win_len = 192
        stride = 96
        min_len = min(len(eeg), len(env_a), len(env_b))
        for start in range(0, min_len - win_len, stride):
            self.dtu_eeg.append(eeg[start:start+win_len])
            self.dtu_aa.append(env_a[start:start+win_len])
            self.dtu_ab.append(env_b[start:start+win_len])
            if len(self.dtu_eeg) >= 200: break

    def _load_kul(self):
        print("Loading KUL Data (S1, First Trial)...")
        mat_path = "/kaggle/input/datasets/lowk1ee/s1-klu/S1_KLU.mat"
        wav_dir = "/kaggle/input/datasets/lowk1ee/audio-klu"
        if not os.path.exists(mat_path):
            print("KUL data not found locally (mocking if needed).")
            return
            
        mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        trials = mat['trials'] if 'trials' in mat else mat['trial']
        trial = trials[0]
        
        fs_eeg = trial.FileHeader.SampleRate
        ch_names = [ch.Label for ch in trial.FileHeader.Channels]
        target_channels = ['Fp1', 'Fp2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8']
        sel_idx = [ch_names.index(tc) if tc in ch_names else [c.upper() for c in ch_names].index(tc.upper()) for tc in target_channels]
        
        kul_eeg_raw = trial.RawData.EegData[:, sel_idx]
        nyq = 0.5 * fs_eeg
        b, a = butter(2, [1.0/nyq, 6.0/nyq], btype='band')
        kul_eeg_raw = filtfilt(b, a, kul_eeg_raw, axis=0)
        kul_eeg_raw = resample(kul_eeg_raw, int(len(kul_eeg_raw) * 64 / fs_eeg), axis=0)
        
        # We need DTU raw for EA computation
        dtu_examples = load_subject_examples(Path("/kaggle/input/datasets/lokeshgile/dataset-eeg/S1_data_preproc.mat") if Path("/kaggle/input/datasets/lokeshgile/dataset-eeg/S1_data_preproc.mat").exists() else REPO_ROOT / "data" / "S1_data_preproc.mat")
        dtu_eeg_raw = dtu_examples[0].eeg[:, [13, 46, 43, 23, 50, 0, 52, 14]]
        dtu_eeg_raw = filtfilt(b, a, dtu_eeg_raw, axis=0)
        
        R_whiten, R_recolor = prepare_alignment_matrices([kul_eeg_raw.T], [dtu_eeg_raw.T])
        R_ea = R_recolor @ R_whiten if R_recolor is not None else R_whiten
        
        kul_eeg_base = normalize_array(kul_eeg_raw)
        kul_eeg_whiten = normalize_array(apply_alignment(kul_eeg_raw.T, R_whiten).T) if R_whiten is not None else kul_eeg_base
        kul_eeg_recolor = normalize_array(apply_alignment(kul_eeg_raw.T, R_ea).T) if R_ea is not None else kul_eeg_base
        
        att_ear = trial.attended_ear
        stimuli = trial.stimuli
        att_wav = stimuli[0] if att_ear == 'L' else stimuli[1]
        unatt_wav = stimuli[1] if att_ear == 'L' else stimuli[0]
        
        def find_wav(name):
            for r, d, f in os.walk(wav_dir):
                if str(name) in f: return os.path.join(r, str(name))
                if str(name)+".wav" in f: return os.path.join(r, str(name)+".wav")
            return None
            
        fs_a, aud_a = wavfile.read(find_wav(att_wav))
        fs_b, aud_b = wavfile.read(find_wav(unatt_wav))
        if len(aud_a.shape) > 1: aud_a = aud_a.mean(axis=1)
        if len(aud_b.shape) > 1: aud_b = aud_b.mean(axis=1)
        
        env_a = normalize_array(extract_28_band_envelope(aud_a, fs_a).T)
        env_b = normalize_array(extract_28_band_envelope(aud_b, fs_b).T)
        
        win_len = 192
        stride = 96
        min_len = min(len(kul_eeg_base), len(env_a), len(env_b))
        for start in range(0, min_len - win_len, stride):
            self.kul_base_eeg.append(kul_eeg_base[start:start+win_len])
            self.kul_ea_eeg.append(kul_eeg_whiten[start:start+win_len])
            self.kul_ea_dtu_eeg.append(kul_eeg_recolor[start:start+win_len])
            self.kul_aa.append(env_a[start:start+win_len])
            self.kul_ab.append(env_b[start:start+win_len])
            if len(self.kul_base_eeg) >= 200: break

    def inspect_layers(self):
        print("--- RUNNING LAYER INSPECTION ---")
        
        def run_batch(dataset_name, eeg_list, aa_list, ab_list):
            self.current_dataset = dataset_name
            with torch.no_grad():
                # [B, T, C] -> [B, C, T]
                eeg_t = torch.FloatTensor(np.array(eeg_list)).permute(0, 2, 1).to(self.device)
                aa_t = torch.FloatTensor(np.array(aa_list)).permute(0, 2, 1).to(self.device)
                ab_t = torch.FloatTensor(np.array(ab_list)).permute(0, 2, 1).to(self.device)
                
                z_eeg, z_a, z_b = self.model(eeg_t, aa_t, ab_t)
                
                # Compute Cosine Similarities
                sim_a = F.cosine_similarity(z_eeg, z_a, dim=1).mean(dim=1).cpu().numpy()
                sim_b = F.cosine_similarity(z_eeg, z_b, dim=1).mean(dim=1).cpu().numpy()
                
                self.activations[dataset_name]["sim_a"] = sim_a
                self.activations[dataset_name]["sim_b"] = sim_b
                self.activations[dataset_name]["margin"] = np.abs(sim_a - sim_b)

        run_batch("DTU", self.dtu_eeg, self.dtu_aa, self.dtu_ab)
        if len(self.kul_base_eeg) > 0:
            run_batch("KUL", self.kul_base_eeg, self.kul_aa, self.kul_ab)
            run_batch("KUL_EA", self.kul_ea_eeg, self.kul_aa, self.kul_ab)
            run_batch("KUL_EA_DTU", self.kul_ea_dtu_eeg, self.kul_aa, self.kul_ab)
            self._analyze_activations()
            
    def _analyze_activations(self):
        print("\n--- RESULTS ---")
        results = []
        layers = ["EEG_Block1", "EEG_Block2", "EEG_Embedding"]
        
        for layer in layers:
            dtu_act = torch.cat(self.activations["DTU"][layer], dim=0).numpy().reshape(200, -1)
            dtu_norm = np.linalg.norm(dtu_act, axis=1).mean()
            dtu_avg_vec = dtu_act.mean(axis=0)
            
            for ds in ["KUL_Base", "KUL_EA", "KUL_EA_DTU"]:
                kul_act = torch.cat(self.activations[ds][layer], dim=0).numpy().reshape(200, -1)
                kul_norm = np.linalg.norm(kul_act, axis=1).mean()
                kul_avg_vec = kul_act.mean(axis=0)
                
                cos_sim = np.dot(dtu_avg_vec, kul_avg_vec) / (np.linalg.norm(dtu_avg_vec) * np.linalg.norm(kul_avg_vec) + 1e-12)
                
                results.append({
                    "Layer": layer,
                    "Dataset": ds,
                    "Norm": kul_norm,
                    "CosineSim_to_DTU": cos_sim
                })
                
            results.append({
                "Layer": layer,
                "Dataset": "DTU",
                "Norm": dtu_norm,
                "CosineSim_to_DTU": 1.0
            })
            
        df = pd.DataFrame(results)
        os.makedirs(REPO_ROOT / "analysis", exist_ok=True)
        df.to_csv(REPO_ROOT / "analysis" / "layer_inspection_after_ea.csv", index=False)
        print(df.to_string())
        
        print("\n[Final Output Matching Metrics]")
        print(f"  DTU Margin Mean: {self.activations['DTU']['margin'].mean():.4f}")
        print(f"  KUL_Base Margin: {self.activations['KUL_Base']['margin'].mean():.4f}")
        print(f"  KUL_EA Margin:   {self.activations['KUL_EA']['margin'].mean():.4f}")
        print(f"  KUL_EA_DTU Marg: {self.activations['KUL_EA_DTU']['margin'].mean():.4f}")
        
        plt.figure(figsize=(10,5))
        sns.kdeplot(self.activations['DTU']['margin'], fill=True, label="DTU")
        sns.kdeplot(self.activations['KUL_Base']['margin'], fill=True, label="KUL_Base")
        sns.kdeplot(self.activations['KUL_EA']['margin'], fill=True, label="KUL_EA")
        sns.kdeplot(self.activations['KUL_EA_DTU']['margin'], fill=True, label="KUL_EA_DTU")
        plt.title("Distribution of Output Margin (sim_a - sim_b)")
        plt.legend()
        plt.savefig("analysis/figures/layer_inspector/margin_distribution_after_ea.png")
        plt.close()

if __name__ == "__main__":
    # Locate best checkpoint
    candidates = []
    
    # 1. Check kaggle inputs for uploaded models (strict match)
    if Path("/kaggle/input").exists():
        for r, d, f in os.walk("/kaggle/input"):
            for file in f:
                if file.endswith('.pth') or file.endswith('.pt'):
                    if 'matchnet' in file.lower() or 'best' in file.lower():
                        candidates.append(os.path.join(r, file))
                        
    # 1b. Check kaggle inputs (loose match - grab any model)
    if not candidates and Path("/kaggle/input").exists():
        for r, d, f in os.walk("/kaggle/input"):
            for file in f:
                if file.endswith('.pth') or file.endswith('.pt'):
                    candidates.append(os.path.join(r, file))
                        
    # 2. Check local repo
    if not candidates:
        for r, d, f in os.walk(REPO_ROOT / "checkpoints"):
            for file in f:
                if 'best' in file.lower() or 'matchnet' in file.lower():
                    candidates.append(os.path.join(r, file))
    
    if not candidates:
        print("ERROR: Could not find checkpoint!")
    else:
        chk_path = candidates[0]
        print(f"Using checkpoint: {chk_path}")
        inspector = EALayerInspector(candidates[0])
        inspector.load_data()
        inspector.inspect_layers()
