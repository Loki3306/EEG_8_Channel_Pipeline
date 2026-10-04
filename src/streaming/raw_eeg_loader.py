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
        for item in chan_arr.ravel():
            while isinstance(item, np.ndarray) and item.size == 1:
                item = item.ravel()[0]
            chan_names.append(str(item).upper())
    elif hasattr(data, "label"):
        for l in data.label.ravel():
            chan_names.append(str(_unwrap_mat_singleton(l)).upper())
    else:
        # Fallback standard BioSemi 64 scalp + 8 EXG + Status
        chan_names = [f"EEG{i+1:02d}" for i in range(64)] + [f"EXG{i+1}" for i in range(8)] + ["STATUS"]
        
    # Find EOG and Scalp Indices
    name_to_idx = {name: idx for idx, name in enumerate(chan_names)}
    
    # Scalp electrodes: first 64 scalp channels (FP1 to O2)
    scalp_indices = [idx for idx, name in enumerate(chan_names) if not name.startswith("EXG") and name != "STATUS"][:64]
    if len(scalp_indices) < 64:
        scalp_indices = list(range(64))
    
    # 3. Continuous Multi-channel Signal
    eeg_obj = data.eeg
    if isinstance(eeg_obj, np.ndarray) and eeg_obj.dtype == object and eeg_obj.size == 1:
        raw_signal = eeg_obj[0, 0].astype(np.float64)
    elif isinstance(eeg_obj, np.ndarray) and eeg_obj.dtype == object and eeg_obj.size > 1:
        # Concatenate trial chunks if segmented
        chunks = [eeg_obj[0, i] for i in range(eeg_obj.shape[1])]
        raw_signal = np.concatenate(chunks, axis=0).astype(np.float64)
    elif isinstance(eeg_obj, np.ndarray) and eeg_obj.dtype != object:
        raw_signal = eeg_obj.astype(np.float64)
    else:
        raw_signal = np.asarray(eeg_obj, dtype=np.float64)
        
    if raw_signal.shape[1] < len(scalp_indices) and raw_signal.shape[0] >= 64:
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
    subject_id = mat_path.stem.split("_")[0]
    mapping = None
    mapping_candidates = [
        Path(__file__).resolve().parents[2] / "scripts" / "verify_baseline" / "data" / "audio_mapping.json",
        Path("/kaggle/working/ISEF_Project/scripts/verify_baseline/data/audio_mapping.json"),
        Path("scripts/verify_baseline/data/audio_mapping.json"),
        Path("data/audio_mapping.json"),
    ]
    for mc in mapping_candidates:
        if mc.exists():
            try:
                import json
                with open(mc, "r", encoding="utf-8") as f:
                    mapping = json.load(f)
                break
            except Exception:
                pass

    # In DTU ActiveTwo raw recordings, triggers are saved in data.event.eeg
    # 140 triggers define 70 intervals:
    # Interval 0: Pre-stimulus baseline (trigger 254)
    # Intervals with single-talker presentation (no competing speaker) are omitted in preprocessed DTU dataset
    # Exactly 60 competing-talker trials are retained:
    COMPETING_RAW_INTERVALS = [
        1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22,
        23, 24, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 41, 42,
        43, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 61, 63, 65,
        66, 67, 68, 69
    ]

    trials: List[RawDTUTrialMetadata] = []
    if hasattr(data, "event") and hasattr(data.event[0, 0], "eeg"):
        event_obj = data.event[0, 0].eeg[0, 0]
        sample_indices = event_obj.sample.ravel() if hasattr(event_obj, "sample") else []
        values = event_obj.value.ravel() if hasattr(event_obj, "value") else []
        
        n_pairs = len(sample_indices) // 2
        if n_pairs >= 70:
            target_intervals = [r for r in COMPETING_RAW_INTERVALS if 2 * r < len(sample_indices)]
        else:
            target_intervals = list(range(n_pairs))
            
        for t_idx, r_idx in enumerate(target_intervals):
            start_s = int(sample_indices[2 * r_idx])
            if 2 * r_idx + 1 < len(sample_indices):
                end_s = int(sample_indices[2 * r_idx + 1])
            else:
                end_s = min(n_samples, start_s + int(50.0 * fs))
                
            trig = int(_unwrap_mat_singleton(values[2 * r_idx])) if 2 * r_idx < len(values) else 0
            
            # Lookup speaker attendance & wav filenames
            m_wav = ""
            f_wav = ""
            att_str = "female" # default
            if mapping and subject_id in mapping:
                t_key = f"trial_{t_idx}"
                t_info = mapping[subject_id].get(t_key, {})
                wav_a = t_info.get("wavA", {}).get("filename", "")
                wav_b = t_info.get("wavB", {}).get("filename", "")
                m_wav = wav_a if "aske" in wav_a.lower() else wav_b
                f_wav = wav_a if "marianne" in wav_a.lower() else wav_b
                att_str = "female" if "marianne" in wav_a.lower() else "male"
                
            trials.append(RawDTUTrialMetadata(
                trial_index=t_idx,
                start_sample=start_s,
                end_sample=end_s,
                attended_speaker=att_str,
                male_wav_name=m_wav,
                female_wav_name=f_wav,
                trigger_code=trig
            ))
            
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


def find_raw_dtu_file(subject_id: str, raw_eeg_dir: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """
    Resolves the path to the raw DTU BioSemi ActiveTwo .mat file for a given subject.
    """
    sub = subject_id.split("_")[0].upper()
    
    candidates = []
    if raw_eeg_dir:
        candidates.append(Path(raw_eeg_dir) / f"{sub}.mat")
        candidates.append(Path(raw_eeg_dir) / f"{sub.lower()}.mat")
        
    candidates.extend([
        Path(f"/kaggle/input/datasets/lokeshgile/dtu-eeg-raw/{sub}.mat"),
        Path(f"/kaggle/input/dtu-eeg-raw/{sub}.mat"),
        Path(f"/kaggle/input/datasets/lokeshgile/raw-s1-dtu/{sub}.mat"),
        Path(f"/kaggle/input/raw-s1-dtu/{sub}.mat"),
        Path(r"C:\Users\lokes\Downloads") / f"{sub}.mat",
        Path("data/raw") / f"{sub}.mat",
    ])
    
    for c in candidates:
        if c.exists():
            return c
            
    # Search recursively in /kaggle/input if available
    if Path("/kaggle/input").exists():
        for p in Path("/kaggle/input").rglob(f"{sub}.mat"):
            return p
        for p in Path("/kaggle/input").rglob(f"{sub.lower()}.mat"):
            return p
            
    return None
