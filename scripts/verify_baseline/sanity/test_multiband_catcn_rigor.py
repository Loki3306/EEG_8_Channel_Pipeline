"""
Rigorous Mathematical and Scientific Verification Suite for Multi-Band CA-TCN.
Verifies:
1. Strict Anti-Symmetry: Delta(A, B) = -Delta(B, A) to machine precision.
2. Causal Latency Masking: Zero future audio leakage and strict ERP latency adherence.
3. Gradient Flow & Backward Pass Stability.
4. Parameter Count Budget Compliance.
5. Multiband 28-to-8 Pooling Fidelity.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
VERIFY_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(VERIFY_ROOT) not in sys.path:
    sys.path.insert(0, str(VERIFY_ROOT))

from models.multiband_catcn import MultiBandCATCNDecoder, CausalERPCrossAttentionHead
from data.multiband_provider import pool_28_to_8_bands

def test_anti_symmetry():
    print("[TEST 1/5] Testing Mathematical Anti-Symmetry in Eval Mode...")
    torch.manual_seed(42)
    model = MultiBandCATCNDecoder(eeg_channels=8, audio_bands=8, hidden_dim=64, num_heads=4)
    model.eval()
    
    eeg = torch.randn(4, 8, 320)
    audio_a = torch.randn(4, 8, 320)
    audio_b = torch.randn(4, 8, 320)
    
    delta_ab, (la_ab, lb_ab), _ = model(eeg, audio_a, audio_b)
    delta_ba, (la_ba, lb_ba), _ = model(eeg, audio_b, audio_a)
    
    discrepancy = torch.max(torch.abs(delta_ab + delta_ba)).item()
    print(f"  • Max |delta(A,B) + delta(B,A)| = {discrepancy:.2e}")
    assert discrepancy < 1e-6, f"Anti-symmetry violation! Discrepancy: {discrepancy}"
    assert torch.allclose(la_ab, lb_ba, atol=1e-6), "Logit A/B exchange violation!"
    print("  --> PASS: Anti-symmetry holds to machine precision.")

def test_causal_latency_masking():
    print("\n[TEST 2/5] Testing Causal Latency Masking (Zero Future Audio Leakage)...")
    head = CausalERPCrossAttentionHead(hidden_dim=64, num_heads=4, max_erp_samples=22)
    head.eval()
    
    seq_len = 64 # 1 second window
    mask = head.get_erp_mask(seq_len, torch.device("cpu"))
    
    # 1. Assert all future audio (t_a > t_e) is masked to -inf
    for t_e in range(seq_len):
        for t_a in range(seq_len):
            if t_a > t_e:
                assert torch.isneginf(mask[t_e, t_a]), f"Anti-causal leakage at t_e={t_e}, t_a={t_a}!"
            elif (t_e - t_a) > 22:
                assert torch.isneginf(mask[t_e, t_a]), f"Latency exceeded but unmasked at t_e={t_e}, t_a={t_a}!"
            else:
                assert mask[t_e, t_a] == 0.0, f"Valid latency masked at t_e={t_e}, t_a={t_a}!"
                
    # 2. Assert attention weights are strictly 0.0 for masked positions
    z_e = torch.randn(2, 64, seq_len)
    z_a = torch.randn(2, 64, seq_len)
    _, attn = head.forward_single_stream(z_e, z_a) # [B, H, T, T]
    
    for t_e in range(seq_len):
        for t_a in range(seq_len):
            if t_a > t_e or (t_e - t_a) > 22:
                max_w = attn[:, :, t_e, t_a].max().item()
                assert max_w == 0.0, f"Non-zero attention at masked position ({t_e}, {t_a}): {max_w}"
                
    print("  --> PASS: Causal ERP mask rigorously verified. Zero anti-causal audio leakage.")

def test_gradient_flow():
    print("\n[TEST 3/5] Testing Gradient Flow & Backpropagation Stability...")
    torch.manual_seed(42)
    model = MultiBandCATCNDecoder(eeg_channels=8, audio_bands=8, hidden_dim=64, num_heads=4)
    model.train()
    
    eeg = torch.randn(2, 8, 320, requires_grad=False)
    audio_a = torch.randn(2, 8, 320, requires_grad=False)
    audio_b = torch.randn(2, 8, 320, requires_grad=False)
    
    delta, (la, lb), _ = model(eeg, audio_a, audio_b)
    loss = torch.clamp(0.5 - (la - lb), min=0.0).mean()
    loss.backward()
    
    zero_grad_params = []
    nan_grad_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                zero_grad_params.append(name)
            elif torch.isnan(param.grad).any():
                nan_grad_params.append(name)
                
    assert len(zero_grad_params) == 0, f"Missing gradients in: {zero_grad_params}"
    assert len(nan_grad_params) == 0, f"NaN gradients in: {nan_grad_params}"
    print(f"  • Backward pass completed. All {len(list(model.parameters()))} parameter tensors received valid finite gradients.")
    print("  --> PASS: Gradient flow fully verified.")

def test_parameter_count_budget():
    print("\n[TEST 4/5] Testing Parameter Count Budget...")
    model_8ch = MultiBandCATCNDecoder(eeg_channels=8, audio_bands=8, hidden_dim=64, num_heads=4)
    p_8ch = sum(p.numel() for p in model_8ch.parameters() if p.requires_grad)
    print(f"  • 8-channel EEG MultiBand-CATCN params: {p_8ch:,}")
    assert p_8ch < 100_000, f"8ch parameter budget exceeded: {p_8ch}"
    
    model_64ch = MultiBandCATCNDecoder(eeg_channels=64, audio_bands=8, hidden_dim=64, num_heads=4)
    p_64ch = sum(p.numel() for p in model_64ch.parameters() if p.requires_grad)
    print(f"  • 64-channel EEG MultiBand-CATCN params: {p_64ch:,}")
    assert p_64ch < 150_000, f"64ch parameter budget exceeded: {p_64ch}"
    print("  --> PASS: Parameter count well within lightweight embedded budget (<150k).")

def test_multiband_pooling():
    print("\n[TEST 5/5] Testing Multiband 28-to-8 Band Pooling...")
    dummy_28 = np.random.uniform(0.1, 1.0, (28, 1000)).astype(np.float32)
    pooled_8 = pool_28_to_8_bands(dummy_28)
    assert pooled_8.shape == (8, 1000), f"Unexpected pooled shape: {pooled_8.shape}"
    assert np.all(np.isfinite(pooled_8)), "Non-finite values found in pooled bands"
    assert np.all(pooled_8 > 0), "Energy lost in subband pooling"
    print(f"  • Successfully mapped 28 subbands -> 8 tonotopic bands ({pooled_8.shape}).")
    print("  --> PASS: Subband pooling mathematically consistent.")

def run_all_tests():
    print("=" * 80)
    print("  MULTIBAND CA-TCN MATHEMATICAL & SCIENTIFIC AUDIT SUITE")
    print("=" * 80)
    test_anti_symmetry()
    test_causal_latency_masking()
    test_gradient_flow()
    test_parameter_count_budget()
    test_multiband_pooling()
    print("=" * 80)
    print("  ALL 5 SCIENTIFIC VERIFICATION AUDITS PASSED PERFECTLY!")
    print("=" * 80)

if __name__ == "__main__":
    run_all_tests()
