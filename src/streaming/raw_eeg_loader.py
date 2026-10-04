"""
Raw DTU BioSemi ActiveTwo EEG and event parser for real-time streaming ingestion.

Parses continuous 512 Hz multi-channel recordings, isolates scalp electrodes (64),
extracts bipolar vertical/horizontal EOG channels, and extracts trial timing triggers.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import scipy.io as sio


@dataclass
class RawDTUTrialMetadata:
    trial_index: int
    start_sample: int
    end_sample: int
    attended_speaker: str # 'male' or 'female', or 'A' / 'B'
    male_wav_name: str
    female_wav_name: str
    trigger_code: int


@dataclass
class RawDTUSubjectData:
    subject_id: str
    fs: float
    channel_names: List[str]
    scalp_indices: List[int]
    eeg_raw: np.ndarray # [N_samples, N_scalp_channels]
    veog_raw: np.ndarray # [N_samples] (EXG3 - EXG5)
    heog_raw: np.ndarray # [N_samples] (EXG4 - EXG7)
    trials: List[RawDTUTrialMetadata]


def _unwrap_mat_singleton(value: Any) -> Any:
    """Recursively unwraps scipy.io 1x1 arrays/structs."""
    current = value
    while isinstance(current, np.ndarray) and current.size == 1:
        current = current[0, 0] if current.ndim == 2 else current.flat[0]
    return current


def load_raw_dtu_file(mat_path: Union[str, Path]) -> RawDTUSubjectData:
    """
    Loads raw continuous BioSemi ActiveTwo EEG recording for a DTU subject.
    
    Parameters:
        mat_path: Path to S<id>.mat raw file.
        
    Returns:
        RawDTUSubjectData containing raw signals, derived EOG, channel indices, and trial markers.
    """
    mat_path = Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(f"Raw DTU EEG file not found: {mat_path}")
        
    mat = sio.loadmat(str(mat_path), struct_as_record=False, squeeze_me=False)
    data = mat["data"][0, 0]
    expinfo = mat["expinfo"][0, 0] if "expinfo" in mat else None
    
    # 1. Sampling Rate
    if hasattr(data, "fsample"):
        fs_obj = data.fsample[0, 0]
        fs = float(fs_obj.eeg[0, 0]) if hasattr(fs_obj, "eeg") else float(fs_obj)
    else:
        fs = 512.0
        
    # 2. Extract Channels & Labels
    chan_names: List[str] = []
    if hasattr(data, "dim") and hasattr(data.dim[0, 0], "chan") and hasattr(data.dim[0, 0].chan[0, 0], "eeg"):
        chan_arr = data.dim[0, 0].chan[0, 0].eeg[0, 0]
        for idx in range(chan_arr.shape[1]):
            val = _unwrap_mat_singleton(chan_arr[0, idx])
            chan_names.append(str(val).upper())
    elif hasattr(data, "label"):
        for l in data.label:
            chan_names.append(str(_unwrap_mat_singleton(l)).upper())
    else:
        # Fallback standard BioSemi 64 scalp + 8 EXG + Status
        chan_names = [f"EEG{i+1:02d}" for i in range(64)] + [f"EXG{i+1}" for i in range(8)] + ["STATUS"]
        
    # Find EOG and Scalp Indices
    name_to_idx = {name: idx for idx, name in enumerate(chan_names)}
    
    # Scalp electrodes: first 64 scalp channels (FP1 to O2)
    scalp_indices = [idx for idx, name in enumerate(chan_names) if not name.startswith("EXG") and name != "STATUS"][:64]
    
    # 3. Continuous Multi-channel Signal
    eeg_obj = data.eeg
    if isinstance(eeg_obj, np.ndarray) and eeg_obj.dtype == object and eeg_obj.size > 1:
        # Concatenate trial chunks if segmented
        chunks = [eeg_obj[0, i] for i in range(eeg_obj.shape[1])]
        raw_signal = np.concatenate(chunks, axis=0)
    elif isinstance(eeg_obj, np.ndarray) and eeg_obj.dtype != object:
        raw_signal = eeg_obj.astype(np.float64)
    else:
        raw_signal = eeg_obj[0, 0].astype(np.float64)
        
    if raw_signal.shape[1] < len(scalp_indices):
        raw_signal = raw_signal.T
        
    # 4. Compute Bipolar EOG
    # VEOG = EXG3 - EXG5 (Vertical eye movements / blinks)
    # HEOG = EXG4 - EXG7 (Horizontal eye movements / saccades)
    idx_exg3 = name_to_idx.get("EXG3", 66)
    idx_exg5 = name_to_idx.get("EXG5", 68)
    idx_exg4 = name_to_idx.get("EXG4", 67)
    idx_exg7 = name_to_idx.get("EXG7", 70)
    
    n_samples = raw_signal.shape[0]
    if raw_signal.shape[1] > max(idx_exg3, idx_exg5):
        veog = (raw_signal[:, idx_exg3] - raw_signal[:, idx_exg5]).astype(np.float64)
    else:
        veog = np.zeros(n_samples, dtype=np.float64)
        
    if raw_signal.shape[1] > max(idx_exg4, idx_exg7):
        heog = (raw_signal[:, idx_exg4] - raw_signal[:, idx_exg7]).astype(np.float64)
    else:
        heog = np.zeros(n_samples, dtype=np.float64)
        
    # 5. Extract Trial Timing & Metadata
    trials: List[RawDTUTrialMetadata] = []
    if expinfo is not None and hasattr(data, "event") and hasattr(data.event[0, 0], "eeg"):
        event_obj = data.event[0, 0].eeg[0, 0]
        sample_indices = event_obj.sample.ravel() if hasattr(event_obj, "sample") else []
        values = event_obj.value.ravel() if hasattr(event_obj, "value") else []
        
        attend_mf = expinfo.attend_mf.ravel() if hasattr(expinfo, "attend_mf") else []
        wav_male = [str(_unwrap_mat_singleton(w)) for w in expinfo.wavfile_male.ravel()] if hasattr(expinfo, "wavfile_male") else []
        wav_female = [str(_unwrap_mat_singleton(w)) for w in expinfo.wavfile_female.ravel()] if hasattr(expinfo, "wavfile_female") else []
        
        # In DTU, every 2 triggers corresponds to 1 audio trial
        n_trials = min(len(attend_mf), len(sample_indices) // 2)
        for t_idx in range(n_trials):
            start_s = int(sample_indices[2 * t_idx])
            if 2 * t_idx + 1 < len(sample_indices):
                end_s = int(sample_indices[2 * t_idx + 1])
            else:
                end_s = min(n_samples, start_s + int(138.0 * fs))
                
            att_code = int(_unwrap_mat_singleton(attend_mf[t_idx]))
            att_str = "male" if att_code == 1 else "female"
            
            m_wav = wav_male[t_idx] if t_idx < len(wav_male) else ""
            f_wav = wav_female[t_idx] if t_idx < len(wav_female) else ""
            trig = int(_unwrap_mat_singleton(values[2 * t_idx])) if 2 * t_idx < len(values) else 0
            
            trials.append(RawDTUTrialMetadata(
                trial_index=t_idx,
                start_sample=start_s,
                end_sample=end_s,
                attended_speaker=att_str,
                male_wav_name=m_wav,
                female_wav_name=f_wav,
                trigger_code=trig
            ))
            
    subject_id = mat_path.stem.split("_")[0]
    return RawDTUSubjectData(
        subject_id=subject_id,
        fs=fs,
        channel_names=chan_names,
        scalp_indices=scalp_indices,
        eeg_raw=raw_signal[:, scalp_indices],
        veog_raw=veog,
        heog_raw=heog,
        trials=trials
    )
