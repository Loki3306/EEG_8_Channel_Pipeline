"""
audio_wav_feature_study.py
==========================
Mini-LOSO experiment: does the two-cutoff audio representation (8 Hz baseline
+ 30 Hz delta/onset) improve short-window AAD accuracy over the standard
28-band gammatone envelope?

Variants tested
───────────────
  Baseline         28 audio channels — existing 8 Hz envelope
  Env+Delta        56 audio channels — baseline + 30 Hz-branch first diff
  Env+Delta+Onset  84 audio channels — baseline + delta + max(delta,0)

Controls
────────
  EEG encoder   : EEGNet (unchanged)
  EEG channels  : 8 standard channels
  Loss          : contrastive (margin=0.05)
  Epochs        : 20 (smoke-test protocol)
  Batch size    : 128
  Train window  : 2s, hop 0.5s

Subjects (Mini-LOSO): S1, S4, S6, S8, S11, S14

Decision gate
─────────────
  avg_gain(2s, 5s) < 1.0%  → KILL
  1.0% – 2.0%              → Optional Full LOSO
  > 2.0%                   → Promote to Full LOSO

Run:
  python training/audio_wav_feature_study.py

GPU optimisations applied vs previous scripts
─────────────────────────────────────────────
  1. Data pre-chunked once per variant (not inside epoch loop)
  2. Evaluation windows batched per trial (one H2D transfer per trial)
  3. Training tensors pre-stacked before DataLoader construction
  4. num_workers=0 (optimal for in-memory TensorDataset)
  5. pin_memory=True for async H2D overlap
"""

import json
import pickle
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from baselines.ridge_aad import load_subject_examples, subject_files
from models.matchnet import ContrastiveMatchNet, contrastive_loss
from training.train_matchnet_loso import (
    butter_bandpass_filter,
    normalize_array,
)

# ── Configuration ──────────────────────────────────────────────────────────────
FS              = 64
TRAIN_WINDOW_SEC = 2.0
TRAIN_HOP_SEC    = 0.5
EPOCHS           = 20
BATCH_SIZE       = 128
LOWCUT           = 1.0
HIGHCUT          = 6.0
NUM_BANDS        = 28
LATENT_DIM       = 64
MARGIN           = 0.05
INPUT_DROPOUT    = 0.2

EEG_CHANNELS = [13, 46, 43, 23, 50, 0, 52, 14]   # 8 standard channels

TARGET_SUBJECTS = ["S1", "S4", "S6", "S8", "S11", "S14"]

# Kaggle paths — fallback to local repo path
_PKL_KAGGLE = Path("/kaggle/working/audio_features_augmented.pkl")
_PKL_LOCAL  = REPO_ROOT / "data" / "audio_features_augmented.pkl"
_MAP_FILE   = REPO_ROOT / "data" / "audio_mapping.json"

# ── Variant definitions ────────────────────────────────────────────────────────
VARIANTS = {
    "Baseline":        {"streams": ["baseline"],              "channels": NUM_BANDS},
    "Env+Delta":       {"streams": ["baseline", "delta"],     "channels": NUM_BANDS * 2},
    "Env+Delta+Onset": {"streams": ["baseline", "delta", "onset"], "channels": NUM_BANDS * 3},
}


# ── Data helpers ───────────────────────────────────────────────────────────────

def load_augmented_features() -> dict:
    pkl = _PKL_KAGGLE if _PKL_KAGGLE.exists() else _PKL_LOCAL
    if not pkl.exists():
        raise FileNotFoundError(
            f"Augmented features not found at {pkl}.\n"
            "Run first:\n"
            "  python data/extract_augmented_envelopes.py "
            "--audio_dir /kaggle/input/datasets/lokeshgile/eeg-audio "
            "--out_file /kaggle/working/audio_features_augmented.pkl"
        )
    print(f"Loading augmented audio features from {pkl} ...")
    with open(pkl, "rb") as f:
        return pickle.load(f)


def assemble_audio(feats_dict: dict, fname: str, streams: list) -> np.ndarray:
    """
    Concatenates the requested streams along axis 0.
    feats_dict[fname] = {"baseline": (28,T), "delta": (28,T), "onset": (28,T)}
    Returns shape: (28*n_streams, T)
    """
    entry = feats_dict[fname]
    return np.concatenate([entry[s] for s in streams], axis=0)


def prepare_subject_data(
    examples,
    subject_stem: str,
    mapping: dict,
    aug_feats: dict,
    streams: list,
) -> tuple:
    """
    Returns (X, YA, YB) as lists of numpy arrays per trial.
    X  : (8, T)
    YA : (C_audio, T) where C_audio = 28 * len(streams)
    YB : (C_audio, T)
    """
    sub_key = subject_stem.replace("_data_preproc", "")
    X, YA, YB = [], [], []

    for i, ex in enumerate(examples):
        # EEG
        eeg = ex.eeg[:, EEG_CHANNELS].T                         # (8, T_eeg)
        eeg = butter_bandpass_filter(eeg, LOWCUT, HIGHCUT, FS, axis=1)
        x   = normalize_array(eeg.T).T                          # (8, T_eeg)

        trial_key = f"trial_{i}"
        if sub_key not in mapping or trial_key not in mapping[sub_key]:
            print(f"  Warning: missing mapping {sub_key}/{trial_key} — skipping")
            continue

        fname_a = mapping[sub_key][trial_key]["wavA"]["filename"]
        fname_b = mapping[sub_key][trial_key]["wavB"]["filename"]

        if fname_a not in aug_feats or fname_b not in aug_feats:
            print(f"  Warning: {fname_a} or {fname_b} not in aug_feats — skipping")
            continue

        env_a = assemble_audio(aug_feats, fname_a, streams)  # (C, T_audio)
        env_b = assemble_audio(aug_feats, fname_b, streams)

        min_len = min(x.shape[1], env_a.shape[1])
        x     = x[:,     :min_len]
        env_a = env_a[:, :min_len]
        env_b = env_b[:, :min_len]

        env_a = normalize_array(env_a.T).T
        env_b = normalize_array(env_b.T).T

        X.append(x)
        YA.append(env_a)
        YB.append(env_b)

    return X, YA, YB


def chunk_all(X, YA, YB, window_sec: float, hop_sec: float):
    """Pre-chunks all trials at once. Called once per variant, not per epoch."""
    win = int(window_sec * FS)
    hop = int(hop_sec    * FS)
    cx, cya, cyb = [], [], []
    for x, ya, yb in zip(X, YA, YB):
        start = 0
        T = x.shape[1]
        while start + win <= T:
            end = start + win
            cx.append(x[:,  start:end])
            cya.append(ya[:, start:end])
            cyb.append(yb[:, start:end])
            start += hop
    return cx, cya, cyb


# ── Evaluation (batch-windowed — one H2D transfer per trial) ──────────────────

def evaluate_batched(model, X, YA, YB, device, window_sec: float) -> tuple:
    """
    Faster evaluation: all windows of a trial are stacked into a single batch
    before the H2D transfer.  Avoids per-window GPU round-trips.
    """
    model.eval()
    win = int(window_sec * FS)
    n_correct = 0.0
    n_total   = 0

    with torch.no_grad():
        for x_np, ya_np, yb_np in zip(X, YA, YB):
            T = x_np.shape[1]
            starts = list(range(0, T - win + 1, win))
            if not starts:
                continue

            # Stack all windows → one GPU transfer
            bx  = np.stack([x_np[:,  s:s+win] for s in starts])  # (W, C_eeg,   win)
            bya = np.stack([ya_np[:, s:s+win] for s in starts])  # (W, C_audio, win)
            byb = np.stack([yb_np[:, s:s+win] for s in starts])

            bx_t  = torch.tensor(bx,  dtype=torch.float32).to(device)
            bya_t = torch.tensor(bya, dtype=torch.float32).to(device)
            byb_t = torch.tensor(byb, dtype=torch.float32).to(device)

            z_eeg, z_a, z_b = model(bx_t, bya_t, byb_t)

            sim_a = F.cosine_similarity(z_eeg, z_a, dim=1).mean(dim=1)  # (W,)
            sim_b = F.cosine_similarity(z_eeg, z_b, dim=1).mean(dim=1)

            correct = (sim_a > sim_b).float() + 0.5 * (sim_a == sim_b).float()
            n_correct += correct.sum().item()
            n_total   += len(starts)

    return n_correct, n_total


# ── Training ───────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_and_eval(
    audio_channels: int,
    X_tr, YA_tr, YB_tr,
    X_va, YA_va, YB_va,
    X_te, YA_te, YB_te,
    device,
) -> tuple:
    """
    Trains a fresh ContrastiveMatchNet for EPOCHS epochs on pre-chunked data.
    Returns (acc_2s, acc_5s) on the held-out test set using the best-val checkpoint.
    """
    # Pre-chunk training data ONCE (not per epoch)
    cx, cya, cyb = chunk_all(X_tr, YA_tr, YB_tr, TRAIN_WINDOW_SEC, TRAIN_HOP_SEC)

    # Stack into tensors — direct cast avoids double allocation
    X_t  = torch.tensor(np.stack(cx),  dtype=torch.float32)
    YA_t = torch.tensor(np.stack(cya), dtype=torch.float32)
    YB_t = torch.tensor(np.stack(cyb), dtype=torch.float32)

    print(f"    Train chunks : {len(cx):5d}  |  Audio tensor : {YA_t.shape}")

    loader = DataLoader(
        TensorDataset(X_t, YA_t, YB_t),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,        # optimal for in-memory TensorDataset
        pin_memory=(device.type == "cuda"),
    )

    model = ContrastiveMatchNet(
        eeg_model_type  = "eegnet",
        eeg_channels    = len(EEG_CHANNELS),
        audio_channels  = audio_channels,
        latent_dim      = LATENT_DIM,
        lags            = [],
        audio_model_type= "standard",
        temporal_pooling= False,
    ).to(device)

    print(f"    Parameters   : {count_parameters(model):,}")

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    use_amp   = device.type == "cuda"
    scaler    = torch.amp.GradScaler("cuda", enabled=use_amp)
    dropout   = nn.Dropout(p=INPUT_DROPOUT)

    best_val  = 0.0
    best_wts  = deepcopy(model.state_dict())

    for epoch in range(EPOCHS):
        model.train()
        for bx, bya, byb in loader:
            bx  = bx.to(device,  non_blocking=True)
            bya = bya.to(device, non_blocking=True)
            byb = byb.to(device, non_blocking=True)

            bx, bya, byb = dropout(bx), dropout(bya), dropout(byb)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                z_eeg, z_a, z_b = model(bx, bya, byb)
                loss, sa, sb = contrastive_loss(z_eeg, z_a, z_b, margin=MARGIN, model=model)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Validate at 2s window
        nc_va, nt_va = evaluate_batched(model, X_va, YA_va, YB_va, device, window_sec=2.0)
        val_acc = nc_va / max(nt_va, 1)

        if val_acc > best_val:
            best_val = val_acc
            best_wts = deepcopy(model.state_dict())

        if (epoch + 1) % 5 == 0:
            print(f"      Epoch {epoch+1:2d}/{EPOCHS} | val_acc={val_acc*100:.2f}%  best={best_val*100:.2f}%")

    # Load best checkpoint for test evaluation
    model.load_state_dict(best_wts)

    nc_2s, nt_2s = evaluate_batched(model, X_te, YA_te, YB_te, device, window_sec=2.0)
    nc_5s, nt_5s = evaluate_batched(model, X_te, YA_te, YB_te, device, window_sec=5.0)

    acc_2s = nc_2s / max(nt_2s, 1)
    acc_5s = nc_5s / max(nt_5s, 1)

    return acc_2s, acc_5s


# ── Main study loop ────────────────────────────────────────────────────────────

def run_study():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Audio WAV Feature Study | device={device}")
    print("=" * 70)

    # Load resources
    with open(_MAP_FILE, "r") as f:
        mapping = json.load(f)

    aug_feats = load_augmented_features()

    all_paths = subject_files()
    if not all_paths:
        raise RuntimeError("No subject files found. Check your data directory.")

    subject_examples = {str(p): load_subject_examples(p) for p in all_paths}

    results = {
        subj: {v: {"2s": None, "5s": None} for v in VARIANTS}
        for subj in TARGET_SUBJECTS
    }

    for held_out_subj in TARGET_SUBJECTS:
        held_out_path = next(
            (p for p in all_paths if p.stem.split("_")[0] == held_out_subj), None
        )
        if held_out_path is None:
            print(f"  Subject {held_out_subj} not found — skipping")
            continue

        train_paths = [p for p in all_paths if p.stem.split("_")[0] != held_out_subj]

        print(f"\n{'─'*70}")
        print(f"Held-out Subject: {held_out_subj}  ({len(train_paths)} training subjects)")
        print(f"{'─'*70}")

        for variant_name, vcfg in VARIANTS.items():
            streams  = vcfg["streams"]
            n_ch     = vcfg["channels"]

            print(f"\n  ▶ Variant: {variant_name}  ({n_ch} audio channels, streams={streams})")

            # Build train / val data
            X_tr, YA_tr, YB_tr = [], [], []
            X_va, YA_va, YB_va = [], [], []

            for p in train_paths:
                X_p, YA_p, YB_p = prepare_subject_data(
                    subject_examples[str(p)], p.stem, mapping, aug_feats, streams
                )
                v_split = max(1, int(0.1 * len(X_p)))
                X_va.extend(X_p[:v_split]);  YA_va.extend(YA_p[:v_split]);  YB_va.extend(YB_p[:v_split])
                X_tr.extend(X_p[v_split:]);  YA_tr.extend(YA_p[v_split:]);  YB_tr.extend(YB_p[v_split:])

            # Build test data
            X_te, YA_te, YB_te = prepare_subject_data(
                subject_examples[str(held_out_path)],
                held_out_path.stem,
                mapping,
                aug_feats,
                streams,
            )

            print(f"    Train trials : {len(X_tr)}  |  Val trials: {len(X_va)}  |  Test trials: {len(X_te)}")

            acc_2s, acc_5s = train_and_eval(
                n_ch,
                X_tr, YA_tr, YB_tr,
                X_va, YA_va, YB_va,
                X_te, YA_te, YB_te,
                device,
            )

            results[held_out_subj][variant_name]["2s"] = acc_2s
            results[held_out_subj][variant_name]["5s"] = acc_5s
            print(f"    Test acc → 2s: {acc_2s*100:.2f}%   5s: {acc_5s*100:.2f}%")

    # ── Results tables ─────────────────────────────────────────────────────────
    vnames = list(VARIANTS.keys())

    def print_table(window: str):
        print(f"\n{'='*80}")
        print(f"PER-SUBJECT ACCURACY  ({window} window)")
        print(f"{'='*80}")
        hdr = "| Subject | " + " | ".join(f"{v:20s}" for v in vnames) + " |"
        sep = "| ------- | " + " | ".join("-"*20 for _ in vnames) + " |"
        print(hdr)
        print(sep)
        for subj in TARGET_SUBJECTS:
            r = results[subj]
            vals = " | ".join(
                f"{r[v][window]*100:20.2f}" if r[v][window] is not None else f"{'N/A':>20}"
                for v in vnames
            )
            print(f"| {subj:7s} | {vals} |")

    print_table("2s")
    print_table("5s")

    # ── Summary / means ────────────────────────────────────────────────────────
    def mean_acc(variant: str, window: str) -> float:
        vals = [
            results[s][variant][window]
            for s in TARGET_SUBJECTS
            if results[s][variant][window] is not None
        ]
        return float(np.mean(vals)) if vals else float("nan")

    print(f"\n{'='*80}")
    print("MEAN ACCURACY AND GAINS vs Baseline")
    print(f"{'='*80}")

    for window in ["2s", "5s"]:
        base_mean = mean_acc("Baseline", window)
        print(f"\n{window} window:")
        print(f"  {'Variant':<25} | {'Mean Acc':>10} | {'Gain':>10}")
        print(f"  {'-'*25}-+-{'-'*10}-+-{'-'*10}")
        for v in vnames:
            m = mean_acc(v, window)
            gain = (m - base_mean) * 100
            print(f"  {v:<25} | {m*100:>10.2f}% | {gain:>+10.2f}%")

    # ── Decision gate ──────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("DECISION GATE  (average gain across 2s and 5s)")
    print(f"{'='*80}")

    base_2 = mean_acc("Baseline", "2s")
    base_5 = mean_acc("Baseline", "5s")

    for v in vnames[1:]:
        g2 = (mean_acc(v, "2s") - base_2) * 100
        g5 = (mean_acc(v, "5s") - base_5) * 100
        avg_g = (g2 + g5) / 2.0

        if avg_g < 1.0:
            decision = "KILL"
        elif avg_g <= 2.0:
            decision = "OPTIONAL FULL LOSO"
        else:
            decision = "PROMOTE TO FULL LOSO"

        print(f"  {v:<25} | avg_gain={avg_g:+.2f}% → {decision}")

    print(f"\n{'='*80}")
    print("Study complete.")


if __name__ == "__main__":
    run_study()
