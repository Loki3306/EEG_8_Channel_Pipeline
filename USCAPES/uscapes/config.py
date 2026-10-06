"""
USCAPES Clinical Suite — Configuration & Global Parameters
"""

from pathlib import Path

# Base Paths
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RAW_EEG_DIR = DATA_DIR / "raw_eeg"
RAW_AUDIO_DIR = DATA_DIR / "raw_audio"
AUDIO_MAPPING_PATH = DATA_DIR / "audio_mapping.json"
STATIC_DIR = PACKAGE_ROOT / "static"

# Sampling Rates
FS_AUDIO = 16000          # 16 kHz Audio Processing
FS_EEG = 64               # 64 Hz Clinical Feature Extraction
FS_RAW_EEG = 512          # 512 Hz BioSemi ActiveTwo Raw Recording Rate

# Chunking & Windowing
AUDIO_BLOCK_SMP = 500     # 31.25 ms at 16 kHz
EEG_BLOCK_SMP = 2         # 31.25 ms at 64 Hz
HOP_SEC = 0.03125         # 31.25 ms
WINDOW_SEC = 5.0          # 5-second sliding decoding window
WINDOW_EEG_SAMPLES = int(WINDOW_SEC * FS_EEG)  # 320 samples

# Electrode Montage (8 Near-Ear Clinical Channels)
MONTAGE_CHANNELS = ["Cz", "FCz", "Fz", "C3", "C4", "CPz", "Pz", "Oz"]
NUM_CHANNELS = len(MONTAGE_CHANNELS)

# BioSemi 64-Channel Hardware Mapping for DTU Dataset
# (Cz, FCz, Fz, C3, C4, CPz, Pz, Oz in 0-indexed BioSemi 64-ch arrangement)
BIOSEMI_DTU_INDICES = [47, 46, 37, 12, 49, 31, 30, 28]

# Acoustic Steering Limits
STEERING_BOOST_DB = 9.0       # Target speaker gain (+9 dB)
STEERING_SUPPRESS_DB = -18.0   # Background interferer attenuation (-18 dB)

# Subject Cohort Registry (Validated DTU 18-Subject Benchmarks)
SUBJECT_COHORT_REGISTRY = {
    "S1": {"acc_5s": 66.3, "acc_10s": 74.6, "acc_20s": 83.3, "win_rate": 87.7, "margin": 28.98, "snr_db": 4.2},
    "S2": {"acc_5s": 67.4, "acc_10s": 78.1, "acc_20s": 85.1, "win_rate": 93.0, "margin": 35.00, "snr_db": 5.1},
    "S3": {"acc_5s": 60.2, "acc_10s": 65.3, "acc_20s": 67.5, "win_rate": 75.4, "margin": 19.00, "snr_db": 2.8},
    "S4": {"acc_5s": 64.5, "acc_10s": 71.5, "acc_20s": 79.8, "win_rate": 91.2, "margin": 22.69, "snr_db": 3.6},
    "S5": {"acc_5s": 65.1, "acc_10s": 73.2, "acc_20s": 80.7, "win_rate": 89.5, "margin": 23.72, "snr_db": 3.8},
    "S6": {"acc_5s": 54.4, "acc_10s": 60.5, "acc_20s": 60.5, "win_rate": 68.4, "margin": 6.56, "snr_db": 1.2},
    "S7": {"acc_5s": 76.0, "acc_10s": 86.0, "acc_20s": 92.1, "win_rate": 98.2, "margin": 47.75, "snr_db": 7.4},
    "S8": {"acc_5s": 72.6, "acc_10s": 85.5, "acc_20s": 93.9, "win_rate": 100.0, "margin": 40.56, "snr_db": 6.5},
    "S9": {"acc_5s": 61.1, "acc_10s": 72.8, "acc_20s": 72.8, "win_rate": 79.0, "margin": 17.91, "snr_db": 2.9},
    "S10": {"acc_5s": 63.2, "acc_10s": 70.6, "acc_20s": 77.2, "win_rate": 73.7, "margin": 24.67, "snr_db": 3.4},
    "S11": {"acc_5s": 57.3, "acc_10s": 61.0, "acc_20s": 62.3, "win_rate": 75.4, "margin": 10.40, "snr_db": 1.8},
    "S12": {"acc_5s": 66.8, "acc_10s": 77.2, "acc_20s": 80.7, "win_rate": 87.7, "margin": 27.74, "snr_db": 4.1},
    "S13": {"acc_5s": 69.8, "acc_10s": 83.3, "acc_20s": 89.5, "win_rate": 94.7, "margin": 37.32, "snr_db": 5.8},
    "S14": {"acc_5s": 66.5, "acc_10s": 76.3, "acc_20s": 77.2, "win_rate": 89.5, "margin": 27.16, "snr_db": 3.9},
    "S15": {"acc_5s": 78.5, "acc_10s": 87.3, "acc_20s": 93.0, "win_rate": 98.2, "margin": 50.77, "snr_db": 7.9},
    "S16": {"acc_5s": 60.7, "acc_10s": 62.7, "acc_20s": 72.8, "win_rate": 77.2, "margin": 18.94, "snr_db": 2.7},
    "S17": {"acc_5s": 64.6, "acc_10s": 73.2, "acc_20s": 80.7, "win_rate": 82.5, "margin": 23.65, "snr_db": 3.5},
    "S18": {"acc_5s": 66.7, "acc_10s": 74.6, "acc_20s": 84.2, "win_rate": 91.2, "margin": 27.22, "snr_db": 4.3},
}
