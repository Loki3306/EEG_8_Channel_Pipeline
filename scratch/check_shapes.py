import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from baselines.ridge_aad import subject_files, load_subject_examples

paths = subject_files()
exs = load_subject_examples(paths[0])
ex = exs[0]
print("eeg shape:", ex.eeg.shape)
print("wav_a shape:", ex.wav_a.shape)
print("wav_b shape:", ex.wav_b.shape)

