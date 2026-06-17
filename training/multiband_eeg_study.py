"""
multiband_eeg_study.py
======================
Mini-LOSO experiment: does multi-band EEG input improve short-window AAD
accuracy compared to the standard 1–6 Hz single-band filter?

Hypothesis
──────────
The current 1–6 Hz bandpass discards theta/alpha/beta EEG information that
encodes attentional state.  For 2–5 s windows, where only 2–8 delta cycles
are available, adding higher-frequency bands may provide complementary
discrimination cues.

Variants
────────
  Baseline          8 ch   1–6 Hz (current pipeline)
  Delta+Theta      16 ch   [1–4 Hz] + [4–8 Hz]
  D+T+Alpha        24 ch   [1–4 Hz] + [4–8 Hz] + [8–12 Hz]
  D+T+A+Beta       32 ch   [1–4 Hz] + [4–8 Hz] + [8–12 Hz] + [12–20 Hz]

Controls
────────
  Audio encoder  : standard AudioEncoder, 28-band gammatone (unchanged)
  Loss           : contrastive (margin=0.05)
  Epochs         : 20
  Batch size     : 128
  Train window   : 2 s, hop 0.5 s

Subjects (Mini-LOSO): S1, S4, S6, S8, S11, S14

Decision gate
─────────────
  avg_gain(2s, 5s) < 1%   → KILL
  1–2%                     → OPTIONAL FULL LOSO
  > 2%                     → PROMOTE

Run:
  python training/multiband_eeg_study.py
"""

import gc
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
    get_mapping_data,
    prepare_dataset,
)

# ── Configuration ──────────────────────────────────────────────────────────────
FS               = 64
TRAIN_WINDOW_SEC = 2.0
TRAIN_HOP_SEC    = 0.5
EPOCHS           = 20
BATCH_SIZE       = 128
NUM_BANDS        = 28        # audio bands (unchanged)
LATENT_DIM       = 64
MARGIN           = 0.05
INPUT_DROPOUT    = 0.2

EEG_CHANNELS = [13, 46, 43, 23, 50, 0, 52, 14]   # 8 standard electrodes

TARGET_SUBJECTS = ["S1", "S4", "S6", "S8", "S11", "S14"]

# ── Band definitions ──────────────────────────────────────────────────────────
BAND_DEFS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta":  (12.0, 20.0),
}

VARIANTS = {
    "Baseline": {
        "bands": [("baseline", 1.0, 6.0)],   # single band, current pipeline
        "n_eeg_ch": len(EEG_CHANNELS),        # 8
    },
    "Delta+Theta": {
        "bands": [("delta", 1.0, 4.0), ("theta", 4.0, 8.0)],
        "n_eeg_ch": len(EEG_CHANNELS) * 2,   # 16
    },
    "D+T+Alpha": {
        "bands": [("delta", 1.0, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 12.0)],
        "n_eeg_ch": len(EEG_CHANNELS) * 3,   # 24
    },
    "D+T+A+Beta": {
        "bands": [("delta", 1.0, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 12.0), ("beta", 12.0, 20.0)],
        "n_eeg_ch": len(EEG_CHANNELS) * 4,   # 32
    },
}


# ── Multi-band EEG preparation ────────────────────────────────────────────────

def prepare_multiband_eeg(examples, bands_spec):
    """
    Band-filter raw EEG into multiple sub-bands and concatenate along the
    channel dimension.  Called once per subject per variant — cached, never
    recomputed inside the epoch loop.

    Parameters
    ----------
    examples : list
        Subject example objects (each has .eeg attribute).
    bands_spec : list of (name, lowcut, highcut)
        Band definitions for this variant.

    Returns
    -------
    X : list of np.ndarray, each shape (n_bands * 8, T)
    """
    X = []
    for ex in examples:
        raw = ex.eeg[:, EEG_CHANNELS].T           # (8, T)
        band_arrays = []
        for _name, lo, hi in bands_spec:
            filtered = butter_bandpass_filter(raw, lo, hi, FS, axis=1)
            normed = normalize_array(filtered.T).T  # per-channel normalisation
            band_arrays.append(normed)
        x = np.concatenate(band_arrays, axis=0)    # (8*n_bands, T)
        X.append(x)
    return X


def prepare_audio(examples, subject_stem, mapping, envelopes):
    """
    Loads the 28-band gammatone envelopes for each trial.
    Identical to baseline pipeline — audio is NOT modified.

    Returns
    -------
    YA, YB : lists of np.ndarray, each shape (28, T)
    valid_indices : list of int — indices of examples that had valid mappings
    """
    sub_key = subject_stem.replace("_data_preproc", "")
    YA, YB = [], []
    valid_indices = []

    for i, ex in enumerate(examples):
        trial_key = f"trial_{i}"
        if sub_key not in mapping or trial_key not in mapping[sub_key]:
            continue
        fname_a = mapping[sub_key][trial_key]["wavA"]["filename"]
        fname_b = mapping[sub_key][trial_key]["wavB"]["filename"]
        if fname_a not in envelopes or fname_b not in envelopes:
            continue

        env_a = envelopes[fname_a]  # (28, T_audio)
        env_b = envelopes[fname_b]

        env_a = normalize_array(env_a.T).T
        env_b = normalize_array(env_b.T).T

        YA.append(env_a)
        YB.append(env_b)
        valid_indices.append(i)

    return YA, YB, valid_indices


def align_eeg_audio(X_raw, YA, YB, valid_indices):
    """
    Selects only the EEG trials that have valid audio mappings and trims
    both to the minimum temporal length.
    """
    X_out, YA_out, YB_out = [], [], []
    for idx, (ya, yb) in zip(valid_indices, zip(YA, YB)):
        x = X_raw[idx]
        min_len = min(x.shape[1], ya.shape[1])
        X_out.append(x[:, :min_len])
        YA_out.append(ya[:, :min_len])
        YB_out.append(yb[:, :min_len])
    return X_out, YA_out, YB_out


# ── Chunking ───────────────────────────────────────────────────────────────────

def chunk_all(X, YA, YB, window_sec, hop_sec):
    """Pre-chunks all trials at once. Called once per variant."""
    win = int(window_sec * FS)
    hop = int(hop_sec * FS)
    cx, cya, cyb = [], [], []
    for x, ya, yb in zip(X, YA, YB):
        start = 0
        T = x.shape[1]
        while start + win <= T:
            end = start + win
            cx.append(x[:, start:end])
            cya.append(ya[:, start:end])
            cyb.append(yb[:, start:end])
            start += hop
    return cx, cya, cyb


# ── Batched evaluation ─────────────────────────────────────────────────────────

def evaluate_batched(model, X, YA, YB, device, window_sec):
    model.eval()
    win = int(window_sec * FS)
    n_correct = 0.0
    n_total = 0

    with torch.no_grad():
        for x_np, ya_np, yb_np in zip(X, YA, YB):
            T = x_np.shape[1]
            starts = list(range(0, T - win + 1, win))
            if not starts:
                continue

            bx  = np.stack([x_np[:, s:s+win]  for s in starts])
            bya = np.stack([ya_np[:, s:s+win] for s in starts])
            byb = np.stack([yb_np[:, s:s+win] for s in starts])

            bx_t  = torch.tensor(bx, dtype=torch.float32).to(device)
            bya_t = torch.tensor(bya, dtype=torch.float32).to(device)
            byb_t = torch.tensor(byb, dtype=torch.float32).to(device)

            z_eeg, z_a, z_b = model(bx_t, bya_t, byb_t)

            sim_a = F.cosine_similarity(z_eeg, z_a, dim=1).mean(dim=1)
            sim_b = F.cosine_similarity(z_eeg, z_b, dim=1).mean(dim=1)

            correct = (sim_a > sim_b).float() + 0.5 * (sim_a == sim_b).float()
            n_correct += correct.sum().item()
            n_total += len(starts)

    return n_correct, n_total


# ── Training ───────────────────────────────────────────────────────────────────

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_and_eval(
    eeg_channels, audio_channels,
    X_tr, YA_tr, YB_tr,
    X_va, YA_va, YB_va,
    X_te, YA_te, YB_te,
    device,
):
    # Pre-chunk training data ONCE
    cx, cya, cyb = chunk_all(X_tr, YA_tr, YB_tr, TRAIN_WINDOW_SEC, TRAIN_HOP_SEC)

    X_t  = torch.tensor(np.stack(cx),  dtype=torch.float32)
    YA_t = torch.tensor(np.stack(cya), dtype=torch.float32)
    YB_t = torch.tensor(np.stack(cyb), dtype=torch.float32)

    print(f"    Train chunks : {len(cx):5d}  |  EEG shape: {X_t.shape}  Audio shape: {YA_t.shape}")

    loader = DataLoader(
        TensorDataset(X_t, YA_t, YB_t),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = ContrastiveMatchNet(
        eeg_model_type="eegnet",
        eeg_channels=eeg_channels,
        audio_channels=audio_channels,
        latent_dim=LATENT_DIM,
        lags=[],
        audio_model_type="standard",
        temporal_pooling=False,
    ).to(device)

    print(f"    Parameters   : {count_params(model):,}")

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    dropout = nn.Dropout(p=INPUT_DROPOUT)

    best_val = 0.0
    best_wts = deepcopy(model.state_dict())

    for epoch in range(EPOCHS):
        model.train()
        for bx, bya, byb in loader:
            bx  = bx.to(device, non_blocking=True)
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

        nc_va, nt_va = evaluate_batched(model, X_va, YA_va, YB_va, device, window_sec=2.0)
        val_acc = nc_va / max(nt_va, 1)

        if val_acc > best_val:
            best_val = val_acc
            best_wts = deepcopy(model.state_dict())

        if (epoch + 1) % 5 == 0:
            print(f"      Epoch {epoch+1:2d}/{EPOCHS} | val_acc={val_acc*100:.2f}%  best={best_val*100:.2f}%")

    model.load_state_dict(best_wts)

    nc_2s, nt_2s = evaluate_batched(model, X_te, YA_te, YB_te, device, window_sec=2.0)
    nc_5s, nt_5s = evaluate_batched(model, X_te, YA_te, YB_te, device, window_sec=5.0)

    acc_2s = nc_2s / max(nt_2s, 1)
    acc_5s = nc_5s / max(nt_5s, 1)

    return acc_2s, acc_5s


# ── Main study loop ────────────────────────────────────────────────────────────

def run_study():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Multi-Band EEG Feature Study | device={device}")
    print("=" * 70)

    mapping, envelopes = get_mapping_data()

    all_paths = subject_files()
    if not all_paths:
        raise RuntimeError("No subject files found.")

    subject_examples = {str(p): load_subject_examples(p) for p in all_paths}

    vnames = list(VARIANTS.keys())
    results = {
        subj: {v: {"2s": None, "5s": None} for v in vnames}
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
        print(f"Held-out: {held_out_subj}  ({len(train_paths)} training subjects)")
        print(f"{'─'*70}")

        for variant_name, vcfg in VARIANTS.items():
            bands_spec = vcfg["bands"]
            n_eeg_ch   = vcfg["n_eeg_ch"]

            print(f"\n  ▶ {variant_name}  ({n_eeg_ch} EEG channels, bands={[b[0] for b in bands_spec]})")

            # ── Prepare training / validation data ─────────────────────────
            X_tr, YA_tr, YB_tr = [], [], []
            X_va, YA_va, YB_va = [], [], []

            for p in train_paths:
                exs = subject_examples[str(p)]

                # Multi-band EEG — computed once per subject
                X_raw = prepare_multiband_eeg(exs, bands_spec)

                # Audio — standard gammatone envelopes (unchanged)
                YA_p, YB_p, valid_idx = prepare_audio(exs, p.stem, mapping, envelopes)

                # Align and trim
                X_p, YA_p, YB_p = align_eeg_audio(X_raw, YA_p, YB_p, valid_idx)

                # 90/10 validation split
                v_split = max(1, int(0.1 * len(X_p)))
                X_va.extend(X_p[:v_split]);   YA_va.extend(YA_p[:v_split]);   YB_va.extend(YB_p[:v_split])
                X_tr.extend(X_p[v_split:]);   YA_tr.extend(YA_p[v_split:]);   YB_tr.extend(YB_p[v_split:])

            # ── Prepare test data ──────────────────────────────────────────
            test_exs = subject_examples[str(held_out_path)]
            X_te_raw = prepare_multiband_eeg(test_exs, bands_spec)
            YA_te, YB_te, valid_te = prepare_audio(test_exs, held_out_path.stem, mapping, envelopes)
            X_te, YA_te, YB_te = align_eeg_audio(X_te_raw, YA_te, YB_te, valid_te)

            print(f"    Train: {len(X_tr)} trials  |  Val: {len(X_va)}  |  Test: {len(X_te)}")

            acc_2s, acc_5s = train_and_eval(
                n_eeg_ch, NUM_BANDS,
                X_tr, YA_tr, YB_tr,
                X_va, YA_va, YB_va,
                X_te, YA_te, YB_te,
                device,
            )

            results[held_out_subj][variant_name]["2s"] = acc_2s
            results[held_out_subj][variant_name]["5s"] = acc_5s
            print(f"    ✓ 2s: {acc_2s*100:.2f}%   5s: {acc_5s*100:.2f}%")

            # Free variant-specific data
            del X_tr, YA_tr, YB_tr, X_va, YA_va, YB_va, X_te, YA_te, YB_te
            gc.collect()

    # ── Results tables ─────────────────────────────────────────────────────────
    def print_table(window):
        print(f"\n{'='*80}")
        print(f"PER-SUBJECT ACCURACY  ({window} window)")
        print(f"{'='*80}")
        hdr = "| Subject | " + " | ".join(f"{v:15s}" for v in vnames) + " |"
        sep = "| ------- | " + " | ".join("-"*15 for _ in vnames) + " |"
        print(hdr)
        print(sep)
        for subj in TARGET_SUBJECTS:
            r = results[subj]
            vals = " | ".join(
                f"{r[v][window]*100:15.2f}" if r[v][window] is not None else f"{'N/A':>15}"
                for v in vnames
            )
            print(f"| {subj:7s} | {vals} |")

    print_table("2s")
    print_table("5s")

    # ── Mean accuracy and gains ────────────────────────────────────────────────
    def mean_acc(variant, window):
        vals = [results[s][variant][window] for s in TARGET_SUBJECTS
                if results[s][variant][window] is not None]
        return float(np.mean(vals)) if vals else float("nan")

    print(f"\n{'='*80}")
    print("MEAN ACCURACY AND GAINS vs Baseline")
    print(f"{'='*80}")

    for window in ["2s", "5s"]:
        base = mean_acc("Baseline", window)
        print(f"\n{window} window:")
        print(f"  {'Variant':<20} | {'Mean Acc':>10} | {'Gain':>10}")
        print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*10}")
        for v in vnames:
            m = mean_acc(v, window)
            gain = (m - base) * 100
            print(f"  {v:<20} | {m*100:>10.2f}% | {gain:>+10.2f}%")

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

        print(f"  {v:<20} | avg_gain={avg_g:+.2f}% → {decision}")

    print(f"\n{'='*80}")
    print("Study complete.")


if __name__ == "__main__":
    run_study()
