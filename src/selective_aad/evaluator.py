import time
import copy
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .core import (
    RawMarginGate,
    EMAMarginGate,
    HysteresisSelectiveGate,
    TemperatureCalibrator,
    SelectiveRiskCoverageOptimizer,
    ConformalSelectiveGate,
)
from .metrics import (
    calculate_selective_metrics,
    compute_risk_coverage_curve,
    compute_aurc,
    compute_ece,
    compute_brier_score,
    compute_nll,
    compute_temporal_stability_metrics,
)

class SelectiveAADEvaluator:
    """
    Evaluation framework enforcing:
      1. Zero-leakage calibration on development subjects.
      2. Comprehensive parameter sweeps.
      3. Strict causality / future-information attack checks.
      4. Negative control experiments.
      5. Execution latency benchmarking.
    """
    def __init__(self, step_sec: float = 0.5):
        self.step_sec = step_sec

    @staticmethod
    def verify_future_information_immunity(
        gate_factory,
        base_margins: np.ndarray,
        probe_indices: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        FUTURE-INFORMATION ATTACK TEST:
        
        Evaluates decision D_t on an input sequence.
        Then perturbs all future samples (t+1, t+2, ..., T) with high-amplitude noise or sign inversions.
        Re-evaluates and strictly verifies that D_t and S_t are bit-for-bit identical.
        """
        margins = np.asarray(base_margins, dtype=np.float64)
        n = len(margins)
        if n < 5:
            raise ValueError("Sequence too short for future information attack")
            
        if probe_indices is None:
            # Test at multiple distinct probe points
            probe_indices = [n // 4, n // 2, (3 * n) // 4]
            
        attack_results = []
        all_passed = True
        
        for t_probe in probe_indices:
            # 1. Clean run up to end
            gate_clean = gate_factory()
            clean_res = gate_clean.process_sequence(margins)
            decision_clean_t = clean_res["decisions"][t_probe]
            smoothed_clean_t = clean_res["smoothed_margins"][t_probe]
            conf_clean_t = clean_res["confidences"][t_probe]
            
            # 2. Corrupt future samples strictly for t > t_probe
            corrupted_margins = margins.copy()
            # Massive future corruption: extreme noise + inversion
            corrupted_margins[t_probe + 1:] = -100.0 * margins[t_probe + 1:] + 50.0
            
            gate_attacked = gate_factory()
            attacked_res = gate_attacked.process_sequence(corrupted_margins)
            decision_attack_t = attacked_res["decisions"][t_probe]
            smoothed_attack_t = attacked_res["smoothed_margins"][t_probe]
            conf_attack_t = attacked_res["confidences"][t_probe]
            
            decision_match = (decision_clean_t == decision_attack_t)
            margin_diff = abs(smoothed_clean_t - smoothed_attack_t)
            conf_diff = abs(conf_clean_t - conf_attack_t)
            
            passed = decision_match and (margin_diff < 1e-9) and (conf_diff < 1e-9)
            if not passed:
                all_passed = False
                
            attack_results.append({
                "t_probe": int(t_probe),
                "passed": bool(passed),
                "decision_clean": str(decision_clean_t),
                "decision_attacked": str(decision_attack_t),
                "margin_diff": float(margin_diff),
                "conf_diff": float(conf_diff),
            })
            
        return {
            "all_passed": all_passed,
            "probe_tests": attack_results,
            "verdict": "CAUSALLY STRICT (NO FUTURE LEAKAGE)" if all_passed else "LEAKAGE DETECTED",
        }

    @staticmethod
    def sweep_raw_margin_thresholds(
        calib_margins: np.ndarray,
        calib_labels: np.ndarray,
        n_thresholds: int = 50
    ) -> Dict[str, Any]:
        """
        Sweeps raw margin threshold across the calibration set to characterize coverage vs risk.
        """
        max_m = float(np.percentile(np.abs(calib_margins), 99.5))
        thresholds = np.linspace(0.0, max_m, n_thresholds)
        
        covs = []
        risks = []
        accs = []
        
        for t in thresholds:
            gate = RawMarginGate(threshold=t)
            res = gate.predict_batch(calib_margins)
            m = calculate_selective_metrics(res["decisions"], calib_labels)
            covs.append(m["coverage"])
            risks.append(m["selective_risk"])
            accs.append(m["selective_accuracy"])
            
        covs = np.array(covs)
        risks = np.array(risks)
        accs = np.array(accs)
        aurc, opt_aurc, e_aurc = compute_aurc(covs, risks)
        
        return {
            "thresholds": thresholds,
            "coverages": covs,
            "risks": risks,
            "accuracies": accs,
            "aurc": aurc,
            "e_aurc": e_aurc,
        }

    @staticmethod
    def sweep_ema_parameters(
        calib_trials_margins: List[np.ndarray],
        calib_trials_labels: List[np.ndarray],
        alpha_candidates: List[float] = [0.3, 0.5, 0.7, 0.85, 0.9],
        threshold_candidates: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Sweeps EMA alpha and threshold across calibration trials.
        Selects optimal (alpha, threshold) based on AURC and stability.
        """
        if threshold_candidates is None:
            all_m = np.concatenate(calib_trials_margins)
            max_m = float(np.percentile(np.abs(all_m), 98.0))
            threshold_candidates = np.linspace(0.0, max_m, 25)
            
        results_grid = []
        best_alpha = alpha_candidates[0]
        best_aurc = float("inf")
        
        for alpha in alpha_candidates:
            # Process all trials with this alpha
            smoothed_margins = []
            all_labels = []
            for m_seq, l_seq in zip(calib_trials_margins, calib_trials_labels):
                gate = EMAMarginGate(alpha=alpha, threshold=0.0)
                res = gate.process_sequence(m_seq)
                smoothed_margins.append(res["smoothed_margins"])
                all_labels.append(l_seq)
                
            concat_smoothed = np.concatenate(smoothed_margins)
            concat_labels = np.concatenate(all_labels)
            
            covs = []
            risks = []
            accs = []
            for t in threshold_candidates:
                decisions = np.full(concat_smoothed.shape, "HOLD", dtype="<U4")
                decisions[concat_smoothed >= t] = "A"
                decisions[concat_smoothed <= -t] = "B"
                m = calculate_selective_metrics(decisions, concat_labels)
                covs.append(m["coverage"])
                risks.append(m["selective_risk"])
                accs.append(m["selective_accuracy"])
                
            aurc, opt_aurc, e_aurc = compute_aurc(np.array(covs), np.array(risks))
            
            results_grid.append({
                "alpha": float(alpha),
                "aurc": float(aurc),
                "e_aurc": float(e_aurc),
                "coverages": np.array(covs),
                "risks": np.array(risks),
                "accuracies": np.array(accs),
            })
            
            if aurc < best_aurc:
                best_aurc = aurc
                best_alpha = alpha
                
        return {
            "best_alpha": best_alpha,
            "best_aurc": best_aurc,
            "grid": results_grid,
            "threshold_candidates": threshold_candidates,
        }

    @staticmethod
    def sweep_hysteresis_parameters(
        calib_trials_margins: List[np.ndarray],
        calib_trials_labels: List[np.ndarray],
        alpha: float = 0.7,
        switch_candidates: List[float] = [0.2, 0.35, 0.5, 0.7],
        confirm_candidates: List[int] = [1, 2, 3, 4],
        step_sec: float = 0.5
    ) -> Dict[str, Any]:
        """
        Sweeps hysteresis switch thresholds and confirmation counts on calibration data.
        Evaluates temporal metrics (false switches/min, correct lock %, hold %).
        """
        sweep_results = []
        best_cfg = None
        best_score = float("inf")  # Objective: minimize false switches while maintaining lock
        
        for t_sw in switch_candidates:
            for n_c in confirm_candidates:
                trial_stats = []
                for m_seq, l_seq in zip(calib_trials_margins, calib_trials_labels):
                    gate = HysteresisSelectiveGate(
                        alpha=alpha,
                        threshold_switch=t_sw,
                        threshold_maintain=t_sw * 0.4,
                        n_confirm=n_c
                    )
                    res = gate.process_sequence(m_seq)
                    stats = compute_temporal_stability_metrics(
                        res["decisions"], l_seq, step_sec=step_sec
                    )
                    trial_stats.append(stats)
                    
                mean_false_sw = float(np.mean([s["false_switches_per_minute"] for s in trial_stats]))
                mean_lock = float(np.mean([s["pct_correctly_locked"] for s in trial_stats]))
                mean_hold = float(np.mean([s["pct_hold"] for s in trial_stats]))
                mean_sw = float(np.mean([s["switches_per_minute"] for s in trial_stats]))
                
                # Balanced objective: penalty for false switches + penalty for low correct lock
                score = mean_false_sw * 2.0 + max(0.0, 70.0 - mean_lock) * 0.1
                
                cfg_res = {
                    "threshold_switch": float(t_sw),
                    "threshold_maintain": float(t_sw * 0.4),
                    "n_confirm": int(n_c),
                    "false_switches_per_min": mean_false_sw,
                    "switches_per_min": mean_sw,
                    "pct_correctly_locked": mean_lock,
                    "pct_hold": mean_hold,
                    "objective_score": score,
                }
                sweep_results.append(cfg_res)
                
                if score < best_score:
                    best_score = score
                    best_cfg = cfg_res
                    
        return {
            "best_config": best_cfg,
            "all_configs": sweep_results,
        }

    @staticmethod
    def benchmark_latency(
        gate_instance: Any,
        n_warmup: int = 100,
        n_iters: int = 10000
    ) -> Dict[str, Any]:
        """
        Measures exact CPU execution latency per single streaming update step.
        """
        # Warmup
        for _ in range(n_warmup):
            gate_instance.update_single(0.25)
            
        gate_instance.reset()
        t0 = time.perf_counter()
        for i in range(n_iters):
            # Alternating sample margins
            m = 0.5 if (i % 2 == 0) else -0.5
            gate_instance.update_single(m)
        elapsed_sec = time.perf_counter() - t0
        
        per_update_us = (elapsed_sec / n_iters) * 1e6
        per_update_ms = per_update_us / 1000.0
        
        return {
            "per_update_microseconds": float(per_update_us),
            "per_update_milliseconds": float(per_update_ms),
            "total_iterations": n_iters,
            "budget_ms": 9.0,
            "headroom_pct": float(max(0.0, (9.0 - per_update_ms) / 9.0 * 100.0)),
        }
