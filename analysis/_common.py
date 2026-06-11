from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.io import loadmat

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]

# Support custom data directory via environment variable or Kaggle path
import os
if "EEG_DATA_DIR" in os.environ:
    DATA_DIR = Path(os.environ["EEG_DATA_DIR"])
elif Path("/kaggle/input").exists():
    try:
        mat_files = list(Path("/kaggle/input").rglob("S*_data_preproc.mat"))
        if mat_files:
            DATA_DIR = mat_files[0].parent
        else:
            DATA_DIR = Path("/kaggle/input")
    except Exception:
        DATA_DIR = Path("/kaggle/input")
else:
    DATA_DIR = Path(r"C:\Users\lokes\Downloads\archive (2)\DATA_preproc")

README_PATH = WORKSPACE_ROOT / "DATA_ANALYSIS.md"
ANALYSIS_DIR = WORKSPACE_ROOT / "analysis"
PLOTS_DIR = ANALYSIS_DIR / "plots"
SUMMARY_DIR = ANALYSIS_DIR / "summaries"


def ensure_output_dirs() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


def subject_files() -> list[Path]:
    return sorted(DATA_DIR.glob("S*_data_preproc.mat"), key=lambda path: int(path.stem.split("_")[0][1:]))


def load_subject_mat(path: Path) -> dict[str, Any]:
    return loadmat(path, squeeze_me=False, struct_as_record=False)


def load_subject_data(path: Path) -> Any:
    mat = load_subject_mat(path)
    return mat["data"][0, 0]


def unwrap_singleton(value: Any) -> Any:
    current = value
    while isinstance(current, np.ndarray) and current.size == 1:
        current = current[0, 0] if current.ndim == 2 else current.flat[0]
    return current


def scalar_int(value: Any) -> int:
    current = unwrap_singleton(value)
    if isinstance(current, np.ndarray):
        return int(np.asarray(current).squeeze())
    return int(current)


def scalar_float(value: Any) -> float:
    current = unwrap_singleton(value)
    if isinstance(current, np.ndarray):
        return float(np.asarray(current).squeeze())
    return float(current)


def object_array_to_list(value: np.ndarray) -> list[Any]:
    return [unwrap_singleton(item) for item in value.ravel()]


def channel_labels(data: Any) -> list[str]:
    chan = data.dim[0, 0].chan[0, 0].eeg[0, 0]
    labels: list[str] = []
    for index in range(chan.shape[1]):
        item = chan[0, index]
        item = unwrap_singleton(item)
        if isinstance(item, np.ndarray):
            labels.append(str(np.asarray(item).squeeze()))
        else:
            labels.append(str(item))
    return labels


def trial_labels(data: Any) -> list[int]:
    events = data.event[0, 0].eeg
    labels: list[int] = []
    for trial_index in range(events.shape[1]):
        event = events[0, trial_index]
        labels.append(scalar_int(event.value[0, 0]))
    return labels


def trial_event_samples(data: Any) -> list[int]:
    events = data.event[0, 0].eeg
    samples: list[int] = []
    for trial_index in range(events.shape[1]):
        event = events[0, trial_index]
        samples.append(scalar_int(event.sample[0, 0]))
    return samples


def fsample_values(data: Any) -> dict[str, int]:
    fsample = data.fsample[0, 0]
    return {
        "eeg": scalar_int(fsample.eeg[0, 0]),
        "wavA": scalar_int(fsample.wavA[0, 0]),
        "wavB": scalar_int(fsample.wavB[0, 0]),
    }


def append_readme_update(lines: Iterable[str], *, title: str | None = None) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    block = [f"[Update - {timestamp}]"]
    if title:
        block.append(f"- {title}")
    for line in lines:
        block.append(f"- {line}")
    content = README_PATH.read_text(encoding="utf-8")
    separator = "\n" if not content.endswith("\n") else ""
    README_PATH.write_text(content + separator + "\n".join(block) + "\n", encoding="utf-8")


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
