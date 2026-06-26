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

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from models.matchnet import ContrastiveMatchNet
from baselines.ridge_aad import load_subject_examples
from preprocessing.euclidean_alignment import prepare_alignment_matrices, apply_alignment

def normalize_array(arr):
    return (arr - arr.mean(axis=0, keepdims=True)) / (arr.std(axis=0, keepdims=True) + 1e-12)

def erb_space(low_freq, high_freq, num_bands):
    erb_low = 21.4 * np.log10(4.37 * low_freq / 1000 + 1)
    erb_high = 21.4 * np.log10(4.37 * high_freq / 1000 + 1)
    erb_points = np.linspace(erb_low, erb_high, num_bands)
    return (10 ** (erb_points / 21.4) - 1) * 1000 / 4.37

def get_erb_bands(cfs):
    bws = 24.7 * (4.37 * cfs / 1000 + 1)
    return cfs - bws / 2, cfs + bws / 2

def apply_bandpass(data, low, high, fs, order=2):
    nyq = 0.5 * fs
    low = max(low / nyq, 0.001)
    high = min(high / nyq, 0.999)
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def extract_28_band_envelope(audio_data, fs_in, fs_out=64, num_bands=28):
    cfs = erb_space(50, 8000, num_bands)
    lows, highs = get_erb_bands(cfs)
    envelopes = []
    for i in range(num_bands):
        band_audio = apply_bandpass(audio_data, lows[i], highs[i], fs_in)
        envelopes.append((np.abs(band_audio) ** 0.3))
    envelopes = np.array(envelopes)
    num_samples_out = int(envelopes.shape[1] * fs_out / fs_in)
    return resample(envelopes, num_samples_out, axis=1)

class EALayerInspector:
    def __init__(self, chk_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ContrastiveMatchNet(eeg_model_type="eegnet", eeg_channels=8, audio_channels=28, latent_dim=64).to(self.device)
        self.model.load_state_dict(torch.load(chk_path, map_location=self.device))
        self.model.eval()
        
        self.datasets = ["DTU", "KUL_Base", "KUL_EA", "KUL_EA_DTU"]
        self.activations = {d: {} for d in self.datasets}
        self.current_dataset = "DTU"
        
        self._register_hooks()
        
        self.data_dict = {d: {"eeg": [], "aa": [], "ab": []} for d in self.datasets}
        
    def _register_hooks(self):
        def get_hook(name):
            def hook(model, input, output):
                if name not in self.activations[self.current_dataset]:
                    self.activations[self.current_dataset][name] = []
                self.activations[self.current_dataset][name].append(output.detach().cpu())
            return hook
            
        self.model.eeg_encoder.block1.register_forward_hook(get_hook("EEG_Block1"))
        self.model.eeg_encoder.block2.register_forward_hook(get_hook("EEG_Block2"))
        self.model.eeg_encoder.output_proj.register_forward_hook(get_hook("EEG_Embedding"))
        
    def load_data(self):
        print("--- LOADING DATA ---")
        
        # Load DTU
        kaggle_path = Path("/kaggle/input/datasets/lokeshgile/dataset-eeg/S1_data_preproc.mat")
        sub_path = kaggle_path if kaggle_path.exists() else REPO_ROOT / "data" / "S1_data_preproc.mat"
        dtu_examples = load_subject_examples(sub_path)
        dtu_channels = [13, 46, 43, 23, 50, 0, 52, 14]
        
        dtu_eegs_raw = []
        for ex in dtu_examples[:2]:
            eeg = ex.eeg[:, dtu_channels]
            nyq = 0.5 * 64
            b, a = butter(2, [1.0/nyq, 6.0/nyq], btype='band')
            dtu_eegs_raw.append(filtfilt(b, a, eeg.T, axis=1))
            
        # Load KUL
        mat_path = "/kaggle/input/datasets/lowk1ee/s1-klu/S1_KLU.mat"
        if not os.path.exists(mat_path): mat_path = str(REPO_ROOT / "data" / "S1_KLU.mat")
        mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        trials = mat['trials'] if 'trials' in mat else mat['data']
        
        kul_eegs_raw = []
        for trial in trials[:2]:
            fs_eeg = 128
            eeg_data = trial.EEG
            if len(eeg_data.shape) > 2: eeg_data = eeg_data[0]
            kul_eeg = eeg_data[:, dtu_channels]
            nyq = 0.5 * fs_eeg
            b, a = butter(2, [1.0/nyq, 6.0/nyq], btype='band')
            kul_eeg = filtfilt(b, a, kul_eeg, axis=0)
            kul_eeg = resample(kul_eeg, int(len(kul_eeg) * 64 / fs_eeg), axis=0).T
            kul_eegs_raw.append(kul_eeg)
            
        # Compute EA Matrices
        R_whiten, R_recolor = prepare_alignment_matrices(kul_eegs_raw, dtu_eegs_raw)
        
        def populate_dataset(name, raw_eegs, R=None):
            # We mock audio since we only care about EEG representations in this script
            for eeg in raw_eegs:
                if R is not None:
                    eeg = apply_alignment(eeg, R)
                eeg_norm = normalize_array(eeg)
                
                win_len = 192
                for start in range(0, eeg_norm.shape[1] - win_len, 96):
                    self.data_dict[name]["eeg"].append(eeg_norm[:, start:start+win_len].T)
                    self.data_dict[name]["aa"].append(np.zeros((win_len, 28)))
                    self.data_dict[name]["ab"].append(np.zeros((win_len, 28)))
                    if len(self.data_dict[name]["eeg"]) >= 200: break
                    
        populate_dataset("DTU", dtu_eegs_raw, None)
        populate_dataset("KUL_Base", kul_eegs_raw, None)
        populate_dataset("KUL_EA", kul_eegs_raw, R_whiten)
        populate_dataset("KUL_EA_DTU", kul_eegs_raw, R_recolor @ R_whiten)
        
    def inspect_layers(self):
        print("--- RUNNING INSPECTION ---")
        for ds in self.datasets:
            self.current_dataset = ds
            eeg_t = torch.FloatTensor(np.array(self.data_dict[ds]["eeg"])).permute(0, 2, 1).to(self.device)
            aa_t = torch.FloatTensor(np.array(self.data_dict[ds]["aa"])).permute(0, 2, 1).to(self.device)
            ab_t = torch.FloatTensor(np.array(self.data_dict[ds]["ab"])).permute(0, 2, 1).to(self.device)
            with torch.no_grad():
                self.model(eeg_t, aa_t, ab_t)
                
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

if __name__ == "__main__":
    candidates = []
    if Path("/kaggle/input").exists():
        for r, d, f in os.walk("/kaggle/input"):
            for file in f:
                if file.endswith('.pth'): candidates.append(os.path.join(r, file))
    for r, d, f in os.walk(REPO_ROOT / "checkpoints"):
        for file in f:
            if file.endswith('.pth'): candidates.append(os.path.join(r, file))
            
    if not candidates:
        print("Model not found!")
    else:
        inspector = EALayerInspector(candidates[0])
        inspector.load_data()
        inspector.inspect_layers()
