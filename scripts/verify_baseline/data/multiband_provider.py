"""
Multi-Band Gammatone Feature Provider and Caching Layer.
Automatically manages 8-band cochlear Gammatone envelopes on Kaggle and local environments.
Supports direct extraction, cached loading, and 28-to-8 band pooling.
"""

from __future__ import annotations
import os
import pickle
import json
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
VERIFY_ROOT = Path(__file__).resolve().parents[1]

# 28-to-8 ERB grouping splits (sum = 28)
GROUP_SPLITS_28_TO_8 = [3, 3, 3, 4, 4, 4, 4, 3]

def pool_28_to_8_bands(env_28: np.ndarray) -> np.ndarray:
    """
    Pools 28 ERB Gammatone subbands into 8 tonotopic bands.
    env_28: shape (28, T)
    Returns: shape (8, T)
    """
    assert env_28.shape[0] == 28, f"Expected 28 bands, got {env_28.shape[0]}"
    pooled = []
    idx = 0
    for count in GROUP_SPLITS_28_TO_8:
        group = env_28[idx:idx + count] # (count, T)
        pooled.append(np.mean(group, axis=0))
        idx += count
    return np.vstack(pooled).astype(np.float32)

def discover_mapping_file() -> Path:
    candidates = [
        Path("/kaggle/input/datasets/lokeshgile/dataset-eeg/audio_mapping.json"),
        Path("/kaggle/input/dataset-eeg/audio_mapping.json"),
        VERIFY_ROOT / "data" / "audio_mapping.json",
        REPO_ROOT / "USCAPES" / "data" / "audio_mapping.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    if Path("/kaggle/input").exists():
        found = list(Path("/kaggle/input").rglob("audio_mapping.json"))
        if found:
            return found[0]
    return candidates[2]

def get_mapping() -> dict:
    map_path = discover_mapping_file()
    if not map_path.exists():
        raise FileNotFoundError(f"audio_mapping.json not found at {map_path}")
    with open(map_path, "r") as f:
        return json.load(f)

def find_cached_envelopes(target_bands: int = 8, custom_path: str = None) -> Path | None:
    if custom_path and Path(custom_path).exists():
        return Path(custom_path)
        
    search_dirs = []
    if os.name != 'nt':
        search_dirs.extend([
            Path("/kaggle/working"),
            Path("/kaggle/input/datasets/lokeshgile/dataset-eeg"),
            Path("/kaggle/input/datasets/lokeshgile/eeg-audio"),
            Path("/kaggle/input"),
        ])
    search_dirs.extend([
        REPO_ROOT / "data",
        VERIFY_ROOT / "data",
    ])
    
    # 1. Look for exact match (e.g. *8band*.pkl)
    exact_pattern = f"*{target_bands}band*.pkl"
    for s_dir in search_dirs:
        if s_dir.exists():
            matches = list(s_dir.rglob(exact_pattern))
            if matches:
                return matches[0]
                
    # 2. Look for any gammatone pickle (e.g. 28-band to pool)
    for s_dir in search_dirs:
        if s_dir.exists():
            matches = list(s_dir.rglob("*gammatone*.pkl"))
            if matches:
                return matches[0]
                
    return None

def get_multiband_envelopes(
    target_bands: int = 8,
    custom_env_file: str = None,
    custom_audio_dir: str = None,
    auto_extract: bool = True
) -> dict[str, np.ndarray]:
    """
    Retrieves or generates 8-band Gammatone envelopes dictionary.
    Keys: WAV filename (e.g. 'aske_story1_trial_1.wav')
    Values: np.ndarray of shape (target_bands, Time)
    """
    cached_path = find_cached_envelopes(target_bands, custom_env_file)
    if cached_path is not None:
        print(f"[DATA PROVIDER] Loading cached Gammatone envelopes from: {cached_path}")
        with open(cached_path, "rb") as f:
            raw_dict = pickle.load(f)
            
        first_key = next(iter(raw_dict.keys()))
        first_val = raw_dict[first_key]
        n_bands = first_val.shape[0] if len(first_val.shape) > 1 else 1
        
        if n_bands == target_bands:
            print(f"[DATA PROVIDER] Envelopes match target dimension: {target_bands} bands.")
            return raw_dict
        elif n_bands == 28 and target_bands == 8:
            print(f"[DATA PROVIDER] Pooling cached 28 subbands into 8 tonotopic ERB subbands...")
            pooled_dict = {}
            for k, v in raw_dict.items():
                pooled_dict[k] = pool_28_to_8_bands(v)
            return pooled_dict
        else:
            print(f"[DATA PROVIDER] Cached envelope dimension ({n_bands}) does not match target ({target_bands}).")
            
    if auto_extract:
        from data.extract_gammatone_envelopes import main as extract_main, discover_audio_directory
        audio_dir = discover_audio_directory(custom_audio_dir)
        if audio_dir.exists() and len(list(audio_dir.glob("*.wav"))) > 0:
            print(f"[DATA PROVIDER] Running self-contained {target_bands}-band Gammatone extraction from {audio_dir}...")
            out_file, envelopes = extract_main(audio_dir, num_bands=target_bands)
            return envelopes
            
    raise FileNotFoundError(
        f"Could not find or extract {target_bands}-band Gammatone envelopes. "
        "Please provide --audio_dir with raw WAV files or --audio_env_file with precomputed PKL."
    )
