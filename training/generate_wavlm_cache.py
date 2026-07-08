import os
import sys
import torch
import numpy as np
import scipy.signal
import scipy.io
import scipy.io.wavfile as wav
from pathlib import Path
from transformers import WavLMModel

def load_wavlm_model(device):
    model = WavLMModel.from_pretrained("microsoft/wavlm-base")
    model.eval()
    model.to(device)
    return model

@torch.no_grad()
def extract_wavlm_features_stereo(audio_l, audio_r, model, device):
    """
    Extracts 768-dim WavLM features for both Left and Right channels simultaneously (Batch Size 2).
    Uses Mixed Precision (FP16) to double GPU efficiency and halve VRAM usage.
    """
    def normalize_audio(audio_data):
        if np.issubdtype(audio_data.dtype, np.integer):
            audio = audio_data.astype(np.float32)
            audio /= np.iinfo(audio_data.dtype).max
        else:
            audio = audio_data.astype(np.float32)
        return audio
        
    audio_l = normalize_audio(audio_l)
    audio_r = normalize_audio(audio_r)
    
    # Batch Left and Right into [2, Time] to double GPU utilization
    audio_tensor = torch.from_numpy(np.stack([audio_l, audio_r])).to(device)
    
    # Use TensorCores via FP16 Autocast
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        outputs = model(audio_tensor)
        hidden_states = outputs.last_hidden_state # [2, TimeWavLM, 768]
    
    # Return native 50Hz features in FP16 to drastically save Disk Space (460MB instead of 2.3GB)
    final_features = hidden_states.cpu().half().clone().detach() # [2, TimeWavLM, 768]
    return final_features[0], final_features[1]

def load_trials_from_raw(mat_path, wav_dir, wavlm_model, device):
    mat = scipy.io.loadmat(mat_path, simplify_cells=True)
    events = mat['EEG_new']['event']
    data_all = mat['EEG_new']['data'] # Shape: (62 trials, 7680 timepoints, 60 channels)
    
    # Bandpass filter the EEG (1-8 Hz)
    b_eeg, a_eeg = scipy.signal.butter(4, [1.0/(128/2), 8.0/(128/2)], btype='band')
    
    trials = []
    
    for trial_idx in range(data_all.shape[0]):
        trial_eeg = data_all[trial_idx] # (7680, 60)
        
        # EEG: Common Average Reference (CAR)
        trial_eeg = trial_eeg - np.mean(trial_eeg, axis=1, keepdims=True)
        
        # EEG: Bandpass Filter
        trial_eeg_filt = scipy.signal.filtfilt(b_eeg, a_eeg, trial_eeg, axis=0)
        
        # EEG: Channel-wise Normalization (Z-score)
        trial_eeg_norm = (trial_eeg_filt - np.mean(trial_eeg_filt, axis=0, keepdims=True)) / (np.std(trial_eeg_filt, axis=0, keepdims=True) + 1e-8)
        eeg_target_length = trial_eeg_norm.shape[0]
        
        # Find Trial ID
        trial_start_event = None
        expected_latency = trial_idx * 7680 + 1
        for ev in events:
            if abs(float(ev[1]) - expected_latency) < 0.5:
                trial_start_event = str(ev[0])
                break
                
        if not trial_start_event or not trial_start_event.isdigit(): continue
            
        audio_id = int(trial_start_event)
        wav_path = os.path.join(wav_dir, f"mixed_{audio_id:03d}.wav")
        if not os.path.exists(wav_path): continue
            
        # Extract WavLM Embeddings efficiently in Stereo (50Hz FP16)
        sr, wav_data = wav.read(wav_path)
        wavlm_l, wavlm_r = extract_wavlm_features_stereo(wav_data[:, 0], wav_data[:, 1], wavlm_model, device)
        
        # Find Switch Points
        epoch_start_lat = trial_idx * 7680 + 1
        switch_points = []
        for ev in events:
            t_str = str(ev[0])
            if t_str in ['179', '184']:
                abs_lat = float(ev[1])
                if epoch_start_lat <= abs_lat < epoch_start_lat + 7680:
                    rel_lat = abs_lat - epoch_start_lat
                    idx_128 = max(0, int(round(rel_lat)))
                    switch_points.append(('L' if t_str == '179' else 'R', idx_128))
                    
        switch_points.sort(key=lambda x: x[1])
        
        # Keep native lengths (EEG is 128Hz, WavLM is 50Hz)
        trials.append({
            'eeg': torch.from_numpy(trial_eeg_norm.T).float(), # Transpose to (60, Time_EEG)
            'wavlm_l': wavlm_l, # [Time_WavLM, 768] (FP16)
            'wavlm_r': wavlm_r,
            'meta': {'switch_points': switch_points}
        })
        
    return trials

if __name__ == "__main__":
    # --- MAIN EXECUTION ---
    data_root = '/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG'
    wav_dir = '/kaggle/input/datasets/lokeshgile/aasd-audio/Stimuli Audio'
    
    import glob
    mat_files = glob.glob(os.path.join(data_root, 'S*', 'S*.mat'))
    mat_files.sort()
    
    cache_dir = Path('/kaggle/working/wavlm_cache')
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading WavLM on {device}...")
    wavlm_model = load_wavlm_model(device)
    
    print(f"Generating WavLM cache directly from WAV files in {cache_dir}...")
    
    # We cannot use Parallel(n_jobs=-1) because loading WavLM on 4 workers will OOM the GPU.
    # We must process sequentially.
    for mat_file in mat_files:
        subj_id = os.path.basename(mat_file).split('.')[0]
        cache_path = cache_dir / f"{subj_id}_wavlm.pt"
        
        if cache_path.exists():
            print(f"  [SKIPPED] {subj_id} is already cached!")
            continue
            
        print(f"  [STARTING] {subj_id}...")
        try:
            trials = load_trials_from_raw(mat_file, wav_dir, wavlm_model, device)
            torch.save({'raw': trials}, cache_path)
            print(f"  [SUCCESS] {subj_id} saved ({len(trials)} trials)")
        except Exception as e:
            print(f"  [ERROR] processing {subj_id}: {e}")
