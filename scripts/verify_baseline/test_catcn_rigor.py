import torch
import numpy as np
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.catcn import (
    CATCNDirectDecoder, CATCN_AudioEncoder, CATCN_EEGEncoder, 
    CrossCorrelationClassificationHead
)

def test_1_causal_leakage():
    """
    Test 1: Causal Leakage Test
    Input x has shape [B, C, T].
    Change only future time steps x[:, :, t0+1:].
    Verify that causal output at all t <= t0 does NOT change.
    """
    print("\n--- TEST 1: Causal Leakage Test (Audio Encoder) ---")
    model = CATCN_AudioEncoder(in_channels=28, hidden_dim=32, dilations=[1, 2, 4, 8, 16])
    model.eval()
    
    T = 200
    t0 = 100
    x_clean = torch.zeros(1, 28, T)
    
    with torch.no_grad():
        out_clean = model(x_clean)
        
        # Perturb strictly future samples: t0+1 to end
        x_perturbed = x_clean.clone()
        x_perturbed[:, :, t0 + 1:] = torch.randn_like(x_perturbed[:, :, t0 + 1:])
        out_perturbed = model(x_perturbed)
        
        # Max difference at t <= t0
        diff_past = (out_clean[:, :, :t0 + 1] - out_perturbed[:, :, :t0 + 1]).abs().max().item()
        diff_future = (out_clean[:, :, t0 + 1:] - out_perturbed[:, :, t0 + 1:]).abs().max().item()
        
    print(f"Max diff at past/present t <= {t0}: {diff_past:.8f}")
    print(f"Max diff at future t > {t0}:        {diff_future:.8f}")
    assert diff_past < 1e-6, f"Causal Leakage Detected! Past output changed by {diff_past}"
    assert diff_future > 1e-4, "Future output unexpectedly unchanged"
    print("PASS: Audio branch is strictly causal (zero future-to-past leakage).")

def test_2_anticausal_leakage():
    """
    Test 2: Anticausal Leakage Test
    Input x has shape [B, C, T].
    Change only past time steps x[:, :, :t0].
    Verify that anticausal output at all t >= t0 does NOT change.
    """
    print("\n--- TEST 2: Anticausal Leakage Test (EEG Encoder) ---")
    model = CATCN_EEGEncoder(in_channels=64, hidden_dim=32, dilations=[1, 2, 4])
    model.eval()
    
    T = 200
    t0 = 100
    x_clean = torch.zeros(1, 64, T)
    
    with torch.no_grad():
        out_clean = model(x_clean)
        
        # Perturb strictly past samples: 0 to t0-1
        x_perturbed = x_clean.clone()
        x_perturbed[:, :, :t0] = torch.randn_like(x_perturbed[:, :, :t0])
        out_perturbed = model(x_perturbed)
        
        # Max difference at t >= t0
        diff_future = (out_clean[:, :, t0:] - out_perturbed[:, :, t0:]).abs().max().item()
        diff_past = (out_clean[:, :, :t0] - out_perturbed[:, :, :t0]).abs().max().item()
        
    print(f"Max diff at future/present t >= {t0}: {diff_future:.8f}")
    print(f"Max diff at past t < {t0}:           {diff_past:.8f}")
    assert diff_future < 1e-6, f"Anticausal Leakage Detected! Future output changed by {diff_future}"
    assert diff_past > 1e-4, "Past output unexpectedly unchanged"
    print("PASS: EEG branch is strictly anticausal (zero past-to-future leakage).")

def test_3_receptive_field():
    """
    Test 3: Receptive Field Verification via Impulse Response
    """
    print("\n--- TEST 3: Receptive Field Impulse Test ---")
    FS = 64.0
    
    # 1. Audio Causal Receptive Field:
    # Expected: RF = 1 + 2*(1+2+4+8+16) = 63 samples = 984.375 ms
    audio_net = CATCN_AudioEncoder(in_channels=28, hidden_dim=16, dilations=[1, 2, 4, 8, 16], dropout=0.0)
    audio_net.eval()
    
    T = 300
    t_impulse = 50
    x_zero = torch.zeros(1, 28, T)
    x_impulse = x_zero.clone()
    x_impulse[:, :, t_impulse] = 1.0 # delta impulse
    
    with torch.no_grad():
        diff_audio = (audio_net(x_impulse) - audio_net(x_zero)).abs().sum(dim=1).squeeze(0) # [T]
        active_indices = torch.where(diff_audio > 0)[0]
        audio_rf_samples = (active_indices.max() - active_indices.min() + 1).item()
        audio_rf_ms = (audio_rf_samples / FS) * 1000.0
        
    print(f"Audio Causal RF: {audio_rf_samples} samples ({audio_rf_ms:.1f} ms) | Active range: {active_indices.min().item()} to {active_indices.max().item()}")
    assert active_indices.min().item() == t_impulse, "Causal impulse response appeared before the impulse!"
    assert audio_rf_samples == 63, f"Expected 63 samples (984 ms), got {audio_rf_samples}"
    print("PASS: Audio branch matches exact paper causal RF (~984 ms).")
    
    # 2. EEG Anticausal Receptive Field:
    # Expected: RF = 1 + 2*(1+2+4) = 15 samples = 234.375 ms
    eeg_net = CATCN_EEGEncoder(in_channels=64, hidden_dim=16, dilations=[1, 2, 4], dropout=0.0)
    eeg_net.eval()
    
    t_impulse = 200
    x_zero = torch.zeros(1, 64, T)
    x_impulse = x_zero.clone()
    x_impulse[:, :, t_impulse] = 1.0
    
    with torch.no_grad():
        diff_eeg = (eeg_net(x_impulse) - eeg_net(x_zero)).abs().sum(dim=1).squeeze(0) # [T]
        active_indices = torch.where(diff_eeg > 0)[0]
        eeg_rf_samples = (active_indices.max() - active_indices.min() + 1).item()
        eeg_rf_ms = (eeg_rf_samples / FS) * 1000.0
        
    print(f"EEG Anticausal RF: {eeg_rf_samples} samples ({eeg_rf_ms:.1f} ms) | Active range: {active_indices.min().item()} to {active_indices.max().item()}")
    assert active_indices.max().item() == t_impulse, "Anticausal impulse response appeared after the impulse!"
    assert eeg_rf_samples == 15, f"Expected 15 samples (234 ms), got {eeg_rf_samples}"
    print("PASS: EEG branch matches exact paper anticausal RF (~234 ms).")

def test_4_cross_correlation_peak():
    """
    Test 4: Cross-Correlation Head with Synthetic Delay
    Verifies that the cross-correlation head captures lag offsets accurately.
    """
    print("\n--- TEST 4: Cross-Correlation Peak Alignment ---")
    head = CrossCorrelationClassificationHead(hidden_dim=4, max_lag_samples=8)
    
    T = 200
    # Generate random audio
    za = torch.randn(1, 4, T)
    
    # Known shift: EEG is delayed by +3 samples relative to audio
    tau_true = 3
    ze = torch.roll(za, shifts=tau_true, dims=-1)
    
    with torch.no_grad():
        # r_all: [B, D * num_lags] -> reshape to [B, D, num_lags]
        r_all = head.compute_cross_correlation(ze, za).view(1, 4, 17)
        # Average across feature channels: [17]
        r_lags = r_all[0].mean(dim=0).numpy()
        
    lags = list(range(-8, 9))
    peak_lag = lags[np.argmax(r_lags)]
    print(f"Known lag: +{tau_true} samples | Peak lag from cross-correlation: {peak_lag:+d} samples (r = {r_lags[lags.index(peak_lag)]:.4f})")
    assert peak_lag == tau_true, f"Expected peak lag at {tau_true}, got {peak_lag}"
    print("PASS: Cross-correlation head identifies temporal lag offset.")

def test_5_ab_anti_symmetry():
    """
    Test 5: A/B Anti-Symmetry Test
    For any input: model(EEG, A, B) -> Delta
                   model(EEG, B, A) -> Delta_swapped
    Must satisfy: Delta_swapped == -Delta with numerical precision.
    """
    print("\n--- TEST 5: A/B Classifier Anti-Symmetry Test ---")
    model = CATCNDirectDecoder(eeg_channels=64, audio_channels=28, hidden_dim=32, max_lag_samples=8)
    model.eval()
    
    eeg = torch.randn(4, 64, 320)
    audio_a = torch.randn(4, 28, 320)
    audio_b = torch.randn(4, 28, 320)
    
    with torch.no_grad():
        delta_normal, (la, lb), _ = model(eeg, audio_a, audio_b)
        delta_swapped, (la_swapped, lb_swapped), _ = model(eeg, audio_b, audio_a)
        
        sum_error = (delta_normal + delta_swapped).abs().max().item()
        
    print(f"Delta (Normal):  {delta_normal[:2].numpy()}")
    print(f"Delta (Swapped): {delta_swapped[:2].numpy()}")
    print(f"Max |Delta + Delta_swapped|: {sum_error:.8e}")
    assert sum_error < 1e-6, f"Anti-symmetry violated! Max error: {sum_error}"
    print("PASS: A/B Anti-Symmetry strictly satisfied (Delta_swapped == -Delta).")

def main():
    print("=" * 60)
    print(" CA-TCN ARCHITECTURAL RIGOR UNIT TEST SUITE")
    print("=" * 60)
    test_1_causal_leakage()
    test_2_anticausal_leakage()
    test_3_receptive_field()
    test_4_cross_correlation_peak()
    test_5_ab_anti_symmetry()
    print("\n" + "=" * 60)
    print(" ALL 5 UNIT TESTS PASSED WITH 100% MATHEMATICAL PRECISION!")
    print("=" * 60)

if __name__ == "__main__":
    main()
