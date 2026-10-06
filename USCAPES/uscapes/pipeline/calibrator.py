"""
USCAPES: Universal Subject-Calibrated Auditory Processing & EEG Steering
Few-Shot 3-Trial Spatial Adaptation & Calibration Engine.

Adapts the 8-electrode spatial projection matrix (SpatialEEGAdapter) and spatial batch-norm
on 3 calibration trials (~3 min) to lock neural attention decoding onto any individual subject.
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from ..config import (
    CHECKPOINT_DIR,
    FS_AUDIO,
    FS_EEG,
    WINDOW_SEC,
    MONTAGE_CHANNELS,
)
from ..models.catcn import CATCNDirectDecoder
from ..models.spatial_adapter import SpatialEEGAdapter
from ..dsp.causal_filters import StreamingCausalEEGFilter
from ..dsp.causal_gammatone import StreamingCausalAudioGammatoneExtractor
from .data_provider import StreamDataProvider


def calibrate_subject(
    subject_id: str,
    backbone_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    calib_trials: int = 3,
    epochs: int = 25,
    lr: float = 1e-3,
    device: Optional[str] = None,
    progress_callback = None
) -> Dict[str, Any]:
    """
    Runs rapid few-shot adaptation for the specified subject across calibration trials (1 to 3).
    
    Returns:
        Summary dict containing elapsed time, final loss, and path to the saved adapted weights.
    """
    start_time = time.perf_counter()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    if output_dir is None:
        output_dir = CHECKPOINT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"catcn_adapted_{subject_id}.pt"

    if backbone_path is None:
        candidates = [
            output_dir / "universal_catcn_backbone.pt",
            CHECKPOINT_DIR / "universal_catcn_backbone.pt",
            Path("checkpoints/universal_catcn_backbone.pt"),
            Path("results/full_cohort/checkpoints/catcn_adapted_reference.pt"),
        ]
        for c in candidates:
            if c.exists():
                backbone_path = c
                break

    print(f"\n{'='*75}")
    print(f"  [USCAPES CALIBRATOR] Calibrating Subject: {subject_id}")
    print(f"  • Calibration Trials: 1 to {calib_trials} (~{calib_trials} minutes)")
    print(f"  • Backbone:           {backbone_path.name if backbone_path else 'Default Architecture'}")
    print(f"  • Target Checkpoint:  {out_path.name}")
    print(f"  • Compute Device:     {dev}")
    print(f"{'='*75}")

    data_provider = StreamDataProvider()
    eeg_filter = StreamingCausalEEGFilter(fs=FS_EEG, lowcut=1.0, highcut=6.0, order=2, n_channels=len(MONTAGE_CHANNELS))
    gamma_a = StreamingCausalAudioGammatoneExtractor(audio_fs=FS_AUDIO, target_fs=FS_EEG)
    gamma_b = StreamingCausalAudioGammatoneExtractor(audio_fs=FS_AUDIO, target_fs=FS_EEG)

    # 1. Ingest Calibration Trials & Extract Features
    X_list, YA_list, YB_list = [], [], []
    win_samples = int(WINDOW_SEC * FS_EEG)  # 320 samples
    hop_samples = int(1.0 * FS_EEG)         # 1.0s hop during calibration for dense data

    for t_id in range(1, calib_trials + 1):
        audio_a, audio_b, raw_eeg, attended = data_provider.load_trial_data(
            subject_id=subject_id, trial_id=t_id, duration_sec=50.0, fs_audio=FS_AUDIO, fs_eeg=FS_EEG
        )
        
        eeg_filter.reset()
        gamma_a.reset()
        gamma_b.reset()

        filt_eeg = eeg_filter.process_chunk(raw_eeg)
        filt_eeg = (filt_eeg - np.mean(filt_eeg, axis=0, keepdims=True)) / (np.std(filt_eeg, axis=0, keepdims=True) + 1e-8)

        env_a = gamma_a.process_chunk(audio_a)
        env_b = gamma_b.process_chunk(audio_b)
        env_a = (env_a - np.mean(env_a)) / (np.std(env_a) + 1e-8)
        env_b = (env_b - np.mean(env_b)) / (np.std(env_b) + 1e-8)

        min_len = min(len(filt_eeg), len(env_a), len(env_b))
        filt_eeg = filt_eeg[:min_len]
        env_a = env_a[:min_len]
        env_b = env_b[:min_len]

        s = 0
        while s + win_samples <= min_len:
            e = s + win_samples
            X_list.append(filt_eeg[s:e].T)                     # [8, 320]
            YA_list.append(env_a[s:e][np.newaxis, :])          # [1, 320]
            YB_list.append(env_b[s:e][np.newaxis, :])          # [1, 320]
            s += hop_samples

    if not X_list:
        raise ValueError(f"No calibration segments extracted for {subject_id}.")

    X_t = torch.from_numpy(np.stack(X_list, axis=0)).float()
    YA_t = torch.from_numpy(np.stack(YA_list, axis=0)).float()
    YB_t = torch.from_numpy(np.stack(YB_list, axis=0)).float()

    dataset = TensorDataset(X_t, YA_t, YB_t)
    loader = DataLoader(dataset, batch_size=min(32, len(dataset)), shuffle=True)
    print(f"  [DATA] Prepared {len(dataset)} calibration windows ({WINDOW_SEC}s each) from Trials 1-{calib_trials}.")

    # 2. Instantiate CA-TCN Model & Adapter
    model = CATCNDirectDecoder(eeg_channels=len(MONTAGE_CHANNELS), audio_channels=1, hidden_dim=64, max_lag_samples=8).to(dev)
    adapter = SpatialEEGAdapter(channels=len(MONTAGE_CHANNELS)).to(dev)

    if backbone_path and backbone_path.exists():
        raw_sd = torch.load(str(backbone_path), map_location=dev, weights_only=False)
        sd = raw_sd["model_state_dict"] if "model_state_dict" in raw_sd else raw_sd
        model.load_state_dict(sd, strict=False)
        print(f"  [MODEL] Loaded universal base weights from {backbone_path.name}")

    # 3. Freeze temporal TCN, adapt ONLY spatial adapter + spatial BN
    for p in model.parameters():
        p.requires_grad = False
    for p in model.eeg_encoder.bn_spatial.parameters():
        p.requires_grad = True
    for p in adapter.parameters():
        p.requires_grad = True

    trainable_params = list(adapter.parameters()) + list(model.eeg_encoder.bn_spatial.parameters())
    num_params = sum(p.numel() for p in trainable_params)
    print(f"  [FREEZE] Temporal TCN frozen. Training {num_params} spatial parameters (Adapter + Spatial BN)...")

    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # 4. Optimization Loop
    model.train()
    adapter.train()
    final_loss = 0.0

    for ep in range(1, epochs + 1):
        ep_loss = 0.0
        n_batches = 0
        for bx, bya, byb in loader:
            bx, bya, byb = bx.to(dev), bya.to(dev), byb.to(dev)
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with spatial adapter
            adapted_bx = adapter(bx)
            delta, (la, lb), _ = model(adapted_bx, bya, byb)
            
            # Contrastive margin loss + Frobenius identity penalty
            loss_margin = torch.clamp(0.5 - (la - lb), min=0.0).mean()
            loss_reg = 0.05 * adapter.identity_regularization_loss()
            loss = loss_margin + loss_reg
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            
            ep_loss += float(loss.item())
            n_batches += 1
            
        scheduler.step()
        final_loss = ep_loss / max(1, n_batches)

        if progress_callback:
            progress_callback(ep, epochs, final_loss)

        if ep % 5 == 0 or ep == epochs:
            print(f"    Epoch {ep:02d}/{epochs:02d} | Loss: {final_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    # 5. Save Adapted Checkpoint
    checkpoint_payload = {
        "model_state_dict": model.state_dict(),
        "adapter_state_dict": adapter.state_dict(),
        "subject_id": subject_id,
        "calib_trials": calib_trials,
        "calibrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "final_loss": float(final_loss)
    }
    torch.save(checkpoint_payload, out_path)
    elapsed = time.perf_counter() - start_time
    print(f"  [SUCCESS] Calibrated checkpoint saved to: {out_path} ({elapsed:.2f}s)")
    print(f"{'='*75}\n")

    return {
        "subject_id": subject_id,
        "calib_trials": calib_trials,
        "epochs": epochs,
        "final_loss": round(float(final_loss), 4),
        "duration_sec": round(elapsed, 2),
        "checkpoint_path": str(out_path),
        "device": str(dev)
    }
