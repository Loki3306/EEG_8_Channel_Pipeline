import unittest
import numpy as np

from src.selective_aad.core import (
    RawMarginGate,
    EMAMarginGate,
    HysteresisSelectiveGate,
    TemperatureCalibrator,
    SelectiveRiskCoverageOptimizer,
    ConformalSelectiveGate,
)
from src.selective_aad.metrics import (
    calculate_selective_metrics,
    compute_risk_coverage_curve,
    compute_aurc,
    compute_ece,
    compute_brier_score,
    compute_temporal_stability_metrics,
)
from src.selective_aad.evaluator import SelectiveAADEvaluator
from src.selective_aad.streaming_gate import SelectiveStreamingGate

class TestSelectiveAAD(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)

    def test_raw_margin_gate(self):
        """Method 1: Verifies RawMarginGate threshold and abstention logic."""
        gate = RawMarginGate(threshold=0.5)
        
        # Test A, B, HOLD
        res_a = gate.predict_single(0.8)
        self.assertEqual(res_a["decision"], "A")
        self.assertTrue(res_a["accepted"])
        self.assertAlmostEqual(res_a["confidence"], 0.8)
        
        res_b = gate.predict_single(-0.7)
        self.assertEqual(res_b["decision"], "B")
        self.assertTrue(res_b["accepted"])
        
        res_hold = gate.predict_single(0.3)
        self.assertEqual(res_hold["decision"], "HOLD")
        self.assertFalse(res_hold["accepted"])
        
        # Batch vectorization equivalence
        batch_margins = np.array([0.8, -0.7, 0.3, -0.2, 1.2])
        batch_res = gate.predict_batch(batch_margins)
        self.assertListEqual(list(batch_res["decisions"]), ["A", "B", "HOLD", "HOLD", "A"])
        self.assertListEqual(list(batch_res["accepted"]), [True, True, False, False, True])

    def test_ema_margin_gate(self):
        """Method 2: Verifies EMA smoothing recursion and streaming vs sequence equivalence."""
        alpha = 0.8
        threshold = 0.4
        gate = EMAMarginGate(alpha=alpha, threshold=threshold)
        
        margins = np.array([0.0, 1.0, 1.0, 1.0, 0.0, -1.0, -1.0])
        seq_res = gate.process_sequence(margins)
        
        # Single update streaming run
        gate.reset()
        stream_smoothed = []
        for m in margins:
            out = gate.update_single(m)
            stream_smoothed.append(out["smoothed_margin"])
            
        np.testing.assert_allclose(seq_res["smoothed_margins"], np.array(stream_smoothed), rtol=1e-6)
        # Smoothing should delay transition from A to B
        self.assertTrue(seq_res["smoothed_margins"][1] < 1.0)
        print(f"[PASS] Method 2 EMA Gate: Smoothed curve = {np.round(seq_res['smoothed_margins'], 2)}")

    def test_hysteresis_selective_gate(self):
        """Method 3: Verifies state machine, consecutive confirmation, and anti-chatter."""
        gate = HysteresisSelectiveGate(
            alpha=0.6,
            threshold_switch=0.4,
            threshold_maintain=0.2,
            n_confirm=2,
            strict_hold_on_deadband=False
        )
        
        # 1. Start in HOLD
        self.assertEqual(gate.current_state, "HOLD")
        
        # First strong positive step: confirmation counter = 1, still HOLD
        res1 = gate.update_single(0.8)
        self.assertEqual(res1["decision"], "HOLD")
        self.assertEqual(res1["confirm_counter"], 1)
        
        # Second strong positive step: confirmation counter = 2 -> Locks onto A!
        res2 = gate.update_single(0.8)
        self.assertEqual(res2["decision"], "A")
        self.assertTrue(res2["switched"])
        
        # 2. Transient negative spike: should NOT switch to B (anti-chatter)
        res3 = gate.update_single(-1.5)
        self.assertEqual(res3["decision"], "A")
        self.assertFalse(res3["switched"])
        
        # 3. Sustained negative input: first confirmation step crossing -threshold_switch
        res4 = gate.update_single(-1.5)
        self.assertEqual(res4["decision"], "A")  # Still in A pending confirmation (confirm_counter=1)
        self.assertEqual(res4["confirm_counter"], 1)
        self.assertFalse(res4["switched"])
        
        # 4. Sustained negative input: second confirmation step -> switches to B!
        res5 = gate.update_single(-1.5)
        self.assertEqual(res5["decision"], "B")
        self.assertTrue(res5["switched"])
        print("[PASS] Method 3 Hysteresis Gate: Anti-chatter and confirmation logic verified.")

    def test_temperature_calibrator(self):
        """Method 4: Verifies post-hoc temperature optimization and probability bounds."""
        calibrator = TemperatureCalibrator()
        
        # Synthetic calibration set: overconfident logits
        calib_margins = np.array([2.5, 3.0, -2.8, -3.2, 1.5, -1.2, 0.4, -0.3, 2.1, -1.9])
        calib_labels = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 0])
        
        fitted_t = calibrator.fit(calib_margins, calib_labels)
        self.assertGreater(fitted_t, 0.0)
        
        p_a, p_b = calibrator.predict_proba(calib_margins)
        np.testing.assert_allclose(p_a + p_b, 1.0, rtol=1e-6)
        self.assertTrue(np.all((p_a >= 0.0) & (p_a <= 1.0)))
        
        # Calibration metrics
        ece_res = compute_ece(p_a, calib_labels, n_bins=5)
        brier = compute_brier_score(p_a, calib_labels)
        self.assertGreaterEqual(ece_res["ece"], 0.0)
        self.assertLessEqual(ece_res["ece"], 1.0)
        self.assertGreaterEqual(brier, 0.0)
        print(f"[PASS] Method 4 Temperature Scaling: Fitted T = {fitted_t:.3f}, ECE = {ece_res['ece']:.4f}, Brier = {brier:.4f}")

    def test_selective_risk_coverage_optimizer(self):
        """Method 5: Verifies Risk-Coverage curve, AURC, and threshold selection."""
        confs = np.array([0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        preds = np.array(["A", "A", "A", "B", "A", "B", "B", "A", "B", "A"])
        gt    = np.array(["A", "A", "A", "B", "B", "B", "A", "B", "A", "B"])
        
        curve = SelectiveRiskCoverageOptimizer.compute_curve(confs, preds, gt)
        self.assertGreater(len(curve["coverages"]), 0)
        self.assertGreaterEqual(curve["aurc"], 0.0)
        self.assertGreaterEqual(curve["e_aurc"], 0.0)
        
        # Test threshold selection for 70% coverage
        tau_70 = SelectiveRiskCoverageOptimizer.select_threshold_for_coverage(confs, 0.70)
        self.assertGreater(tau_70, 0.0)
        print(f"[PASS] Method 5 Risk-Coverage: AURC = {curve['aurc']:.4f}, E-AURC = {curve['e_aurc']:.4f}, Tau@70% = {tau_70:.2f}")

    def test_conformal_selective_gate(self):
        """Method 6: Verifies conformal prediction sets and assumption audit."""
        gate = ConformalSelectiveGate(error_rate_target=0.15)
        
        self.assertIn("window_level_exchangeability", gate.assumption_audit)
        self.assertFalse(gate.assumption_audit["window_level_exchangeability"])
        
        # Synthetic calibration
        calib_probs = np.array([0.95, 0.90, 0.85, 0.80, 0.75, 0.60, 0.15, 0.10, 0.05, 0.02])
        calib_labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        
        q = gate.fit_calibration_quantile(calib_probs, calib_labels)
        self.assertGreater(q, 0.0)
        
        # Test predictions
        test_probs = np.array([0.98, 0.01, 0.51])  # Confident A, Confident B, Ambiguous
        sets_res = gate.predict_sets(test_probs)
        
        self.assertEqual(sets_res["decisions"][0], "A")
        self.assertEqual(sets_res["decisions"][1], "B")
        self.assertEqual(sets_res["decisions"][2], "HOLD")  # Ambiguous -> HOLD
        print(f"[PASS] Method 6 Conformal Gate: Calibrated Quantile = {q:.3f}, Sets: {list(sets_res['decisions'])}")

    def test_future_information_attack_immunity(self):
        """Integrity Test: Verifies that decisions at time t are 100% immune to future information."""
        np.random.seed(123)
        margins = np.random.randn(200)
        
        # Test EMA Gate
        res_ema = SelectiveAADEvaluator.verify_future_information_immunity(
            lambda: EMAMarginGate(alpha=0.7, threshold=0.25),
            margins
        )
        self.assertTrue(res_ema["all_passed"], f"EMA gate failed future information test: {res_ema}")
        
        # Test Hysteresis Gate
        res_hyst = SelectiveAADEvaluator.verify_future_information_immunity(
            lambda: HysteresisSelectiveGate(alpha=0.7, threshold_switch=0.35, threshold_maintain=0.15, n_confirm=2),
            margins
        )
        self.assertTrue(res_hyst["all_passed"], f"Hysteresis gate failed future information test: {res_hyst}")
        print("[PASS] Strict Causality Verified: Future information attack immunity confirmed bit-for-bit.")

    def test_streaming_gate_latency_budget(self):
        """Performance Test: Verifies CPU latency per update is < 0.1 ms (9 ms budget headroom > 98%)."""
        gate = SelectiveStreamingGate(alpha=0.7, threshold_switch=0.35, threshold_maintain=0.15, n_confirm=2)
        bench = SelectiveAADEvaluator.benchmark_latency(gate, n_warmup=100, n_iters=10000)
        
        self.assertLess(bench["per_update_milliseconds"], 0.1, f"Latency too high: {bench['per_update_milliseconds']} ms")
        self.assertGreater(bench["headroom_pct"], 98.0)
        print(f"[PASS] Latency Benchmark: {bench['per_update_microseconds']:.2f} µs/update ({bench['per_update_milliseconds']:.4f} ms) | Budget headroom: {bench['headroom_pct']:.2f}%")

if __name__ == "__main__":
    unittest.main()
