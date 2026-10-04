import math
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from scipy import optimize

def _safe_trapezoid(y, x):
    """NumPy 2.0+ compatible trapezoidal integration."""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    elif hasattr(np, "trapz"):
        return np.trapz(y, x)
    else:
        from scipy import integrate
        return integrate.trapezoid(y, x)

class RawMarginGate:
    """
    Method 1: Raw Signed Margin Gate.
    
    Consumes signed neural margin m_t = logit_A - logit_B from frozen CA-TCN.
    Confidence score is defined as c_t = |m_t|.
    
    Decision Rule:
      - m_t >= +threshold  => 'A'
      - m_t <= -threshold  => 'B'
      - otherwise          => 'HOLD'
    """
    def __init__(self, threshold: float = 0.0):
        if threshold < 0.0:
            raise ValueError(f"Threshold must be non-negative, got {threshold}")
        self.threshold = float(threshold)

    def set_threshold(self, threshold: float):
        if threshold < 0.0:
            raise ValueError(f"Threshold must be non-negative, got {threshold}")
        self.threshold = float(threshold)

    def predict_single(self, margin: float) -> Dict[str, Any]:
        """Processes a single scalar margin."""
        m = float(margin)
        conf = abs(m)
        if m >= self.threshold:
            decision = "A"
            accepted = True
        elif m <= -self.threshold:
            decision = "B"
            accepted = True
        else:
            decision = "HOLD"
            accepted = False
        return {
            "decision": decision,
            "accepted": accepted,
            "confidence": conf,
            "margin": m,
            "is_hold": (decision == "HOLD"),
        }

    def predict_batch(self, margins: np.ndarray) -> Dict[str, np.ndarray]:
        """Vectorized prediction across an array of margins."""
        m = np.asarray(margins, dtype=np.float64)
        conf = np.abs(m)
        
        decisions = np.full(m.shape, "HOLD", dtype="<U4")
        accepted = conf >= self.threshold
        
        decisions[(m >= self.threshold)] = "A"
        decisions[(m <= -self.threshold)] = "B"
        
        return {
            "decisions": decisions,
            "accepted": accepted,
            "confidences": conf,
            "margins": m,
            "is_hold": (decisions == "HOLD"),
        }


class EMAMarginGate:
    """
    Method 2: Temporally Smoothed Margin (Streaming EMA).
    
    Applies streaming exponential moving average on raw margin:
      S_t = alpha * S_{t-1} + (1 - alpha) * m_t
    
    Confidence score is c_t = |S_t|.
    
    Decision Rule:
      - S_t >= +threshold  => 'A'
      - S_t <= -threshold  => 'B'
      - otherwise          => 'HOLD'
    """
    def __init__(self, alpha: float = 0.7, threshold: float = 0.0):
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"Alpha must be in [0.0, 1.0], got {alpha}")
        if threshold < 0.0:
            raise ValueError(f"Threshold must be non-negative, got {threshold}")
        self.alpha = float(alpha)
        self.threshold = float(threshold)
        self.smoothed_margin = 0.0
        self.initialized = False

    def reset(self):
        self.smoothed_margin = 0.0
        self.initialized = False

    def update_single(self, margin: float) -> Dict[str, Any]:
        """Processes single streaming update."""
        m = float(margin)
        if not self.initialized:
            self.smoothed_margin = m
            self.initialized = True
        else:
            self.smoothed_margin = self.alpha * self.smoothed_margin + (1.0 - self.alpha) * m
            
        conf = abs(self.smoothed_margin)
        if self.smoothed_margin >= self.threshold:
            decision = "A"
            accepted = True
        elif self.smoothed_margin <= -self.threshold:
            decision = "B"
            accepted = True
        else:
            decision = "HOLD"
            accepted = False
            
        return {
            "decision": decision,
            "accepted": accepted,
            "confidence": conf,
            "raw_margin": m,
            "smoothed_margin": float(self.smoothed_margin),
            "is_hold": (decision == "HOLD"),
        }

    def process_sequence(self, margins: np.ndarray) -> Dict[str, np.ndarray]:
        """Offline sequence processing with exact causal temporal recursion."""
        m = np.asarray(margins, dtype=np.float64)
        n = len(m)
        smoothed = np.zeros(n, dtype=np.float64)
        if n == 0:
            return {
                "decisions": np.array([], dtype="<U4"),
                "accepted": np.array([], dtype=bool),
                "confidences": np.array([], dtype=np.float64),
                "smoothed_margins": smoothed,
                "is_hold": np.array([], dtype=bool),
            }
            
        s = m[0]
        smoothed[0] = s
        for i in range(1, n):
            s = self.alpha * s + (1.0 - self.alpha) * m[i]
            smoothed[i] = s
            
        conf = np.abs(smoothed)
        decisions = np.full(n, "HOLD", dtype="<U4")
        accepted = conf >= self.threshold
        decisions[smoothed >= self.threshold] = "A"
        decisions[smoothed <= -self.threshold] = "B"
        
        return {
            "decisions": decisions,
            "accepted": accepted,
            "confidences": conf,
            "smoothed_margins": smoothed,
            "is_hold": (decisions == "HOLD"),
        }


class HysteresisSelectiveGate:
    """
    Method 3: Dual-Threshold Hysteresis + Consecutive-Confirmation Selective Gate.
    
    A stateful decision controller designed to prevent volume chatter and erratic switching.
    States: 'A', 'B', 'HOLD'
    
    State transition logic:
      - Switching from A -> B requires smoothed score S_t < -threshold_switch for N consecutive steps.
      - Switching from B -> A requires smoothed score S_t > +threshold_switch for N consecutive steps.
      - If locked in A or B, but evidence drops inside deadband (|S_t| < threshold_maintain):
        If strict_hold_on_deadband is True, drops into HOLD.
        Otherwise retains current state until affirmative switch.
      - Locking on from HOLD into A requires S_t >= +threshold_switch for N consecutive steps.
      - Locking on from HOLD into B requires S_t <= -threshold_switch for N consecutive steps.
    """
    def __init__(
        self,
        alpha: float = 0.7,
        threshold_switch: float = 0.35,
        threshold_maintain: float = 0.15,
        n_confirm: int = 2,
        strict_hold_on_deadband: bool = True,
    ):
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"Alpha must be in [0.0, 1.0], got {alpha}")
        if threshold_switch < threshold_maintain:
            raise ValueError(f"threshold_switch ({threshold_switch}) must be >= threshold_maintain ({threshold_maintain})")
        if n_confirm < 1:
            raise ValueError(f"n_confirm must be >= 1, got {n_confirm}")
            
        self.alpha = float(alpha)
        self.threshold_switch = float(threshold_switch)
        self.threshold_maintain = float(threshold_maintain)
        self.n_confirm = int(n_confirm)
        self.strict_hold_on_deadband = bool(strict_hold_on_deadband)
        
        # State variables
        self.smoothed_margin = 0.0
        self.initialized = False
        self.current_state = "HOLD"  # 'A', 'B', 'HOLD'
        self.pending_target = None
        self.confirm_counter = 0

    def reset(self):
        self.smoothed_margin = 0.0
        self.initialized = False
        self.current_state = "HOLD"
        self.pending_target = None
        self.confirm_counter = 0

    def update_single(self, margin: float) -> Dict[str, Any]:
        """Causal streaming update for a single step."""
        m = float(margin)
        if not self.initialized:
            self.smoothed_margin = m
            self.initialized = True
        else:
            self.smoothed_margin = self.alpha * self.smoothed_margin + (1.0 - self.alpha) * m
            
        s = self.smoothed_margin
        conf = abs(s)
        switched = False
        prev_state = self.current_state
        
        # Determine candidate direction
        if s >= self.threshold_switch:
            candidate = "A"
        elif s <= -self.threshold_switch:
            candidate = "B"
        elif abs(s) < self.threshold_maintain:
            candidate = "HOLD"
        else:
            # Deadband region between maintain and switch
            candidate = "DEADBAND"
            
        if self.current_state == "HOLD":
            if candidate in ["A", "B"]:
                if self.pending_target == candidate:
                    self.confirm_counter += 1
                else:
                    self.pending_target = candidate
                    self.confirm_counter = 1
                    
                if self.confirm_counter >= self.n_confirm:
                    self.current_state = candidate
                    switched = True
                    self.pending_target = None
                    self.confirm_counter = 0
            else:
                self.pending_target = None
                self.confirm_counter = 0
                
        elif self.current_state == "A":
            if candidate == "B":
                if self.pending_target == "B":
                    self.confirm_counter += 1
                else:
                    self.pending_target = "B"
                    self.confirm_counter = 1
                    
                if self.confirm_counter >= self.n_confirm:
                    self.current_state = "B"
                    switched = True
                    self.pending_target = None
                    self.confirm_counter = 0
            elif candidate == "HOLD" and self.strict_hold_on_deadband:
                self.current_state = "HOLD"
                switched = True
                self.pending_target = None
                self.confirm_counter = 0
            else:
                # Remains in A
                if candidate == "A":
                    self.pending_target = None
                    self.confirm_counter = 0
                    
        elif self.current_state == "B":
            if candidate == "A":
                if self.pending_target == "A":
                    self.confirm_counter += 1
                else:
                    self.pending_target = "A"
                    self.confirm_counter = 1
                    
                if self.confirm_counter >= self.n_confirm:
                    self.current_state = "A"
                    switched = True
                    self.pending_target = None
                    self.confirm_counter = 0
            elif candidate == "HOLD" and self.strict_hold_on_deadband:
                self.current_state = "HOLD"
                switched = True
                self.pending_target = None
                self.confirm_counter = 0
            else:
                # Remains in B
                if candidate == "B":
                    self.pending_target = None
                    self.confirm_counter = 0

        accepted = (self.current_state in ["A", "B"])
        return {
            "decision": self.current_state,
            "accepted": accepted,
            "switched": switched,
            "prev_state": prev_state,
            "confidence": conf,
            "raw_margin": m,
            "smoothed_margin": float(s),
            "is_hold": (self.current_state == "HOLD"),
            "confirm_counter": self.confirm_counter,
        }

    def process_sequence(self, margins: np.ndarray) -> Dict[str, np.ndarray]:
        """Processes full sequence with state reset at start."""
        self.reset()
        m = np.asarray(margins, dtype=np.float64)
        n = len(m)
        decisions = []
        accepted = []
        switches = []
        confs = []
        smoothed = []
        
        for val in m:
            res = self.update_single(val)
            decisions.append(res["decision"])
            accepted.append(res["accepted"])
            switches.append(res["switched"])
            confs.append(res["confidence"])
            smoothed.append(res["smoothed_margin"])
            
        return {
            "decisions": np.array(decisions, dtype="<U4"),
            "accepted": np.array(accepted, dtype=bool),
            "switches": np.array(switches, dtype=bool),
            "confidences": np.array(confs, dtype=np.float64),
            "smoothed_margins": np.array(smoothed, dtype=np.float64),
            "is_hold": (np.array(decisions) == "HOLD"),
        }


class TemperatureCalibrator:
    """
    Method 4: Post-Hoc Temperature Calibration.
    
    Transforms signed neural margin m = logit_A - logit_B into calibrated posterior:
      p(A) = sigmoid(m / T) = 1 / (1 + exp(-m / T))
      p(B) = 1 - p(A)
      
    Scalar T > 0 is fitted strictly on calibration data minimizing negative log-likelihood (NLL).
    The frozen CA-TCN weights are untouched.
    """
    def __init__(self, temperature: float = 1.0):
        if temperature <= 0.0:
            raise ValueError(f"Temperature must be strictly positive, got {temperature}")
        self.temperature = float(temperature)

    def fit(self, calib_margins: np.ndarray, calib_labels: np.ndarray, bounds: Tuple[float, float] = (0.01, 20.0)) -> float:
        """
        Fits optimal temperature scalar on calibration margins and binary labels.
        Labels: 1 for Stream A (ground truth), 0 for Stream B.
        """
        margins = np.asarray(calib_margins, dtype=np.float64)
        labels = np.asarray(calib_labels, dtype=np.float64)
        
        if len(margins) == 0:
            raise ValueError("Cannot fit temperature on empty calibration array")
            
        def nll_obj(t_val: float) -> float:
            # Scaled margin
            scaled_m = margins / max(t_val, 1e-4)
            # Numerically stable BCE: log(1 + exp(-scaled_m)) for y=1, log(1 + exp(scaled_m)) for y=0
            # log(1 + exp(-x)) = max(0, -x) + log(1 + exp(-|x|))
            loss = np.where(
                labels == 1.0,
                np.log1p(np.exp(-np.abs(scaled_m))) + np.maximum(-scaled_m, 0.0),
                np.log1p(np.exp(-np.abs(scaled_m))) + np.maximum(scaled_m, 0.0)
            )
            return float(np.mean(loss))
            
        res = optimize.minimize_scalar(nll_obj, bounds=bounds, method="bounded")
        self.temperature = float(res.x)
        return self.temperature

    def predict_proba(self, margins: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns calibrated (p(A), p(B)).
        """
        m = np.asarray(margins, dtype=np.float64)
        scaled_m = np.clip(m / self.temperature, -50.0, 50.0)
        p_a = 1.0 / (1.0 + np.exp(-scaled_m))
        p_b = 1.0 - p_a
        return p_a, p_b

    def get_confidence(self, margins: np.ndarray) -> np.ndarray:
        """
        Confidence in [0.0, 1.0]: 2 * |p(A) - 0.5| = |p(A) - p(B)|.
        """
        p_a, p_b = self.predict_proba(margins)
        return np.abs(p_a - p_b)

    def predict_selective(self, margins: np.ndarray, prob_threshold: float = 0.5) -> Dict[str, np.ndarray]:
        """
        Selective decision based on calibrated posterior threshold tau in [0.5, 1.0].
        If p(A) >= tau => 'A'
        If p(B) >= tau => 'B'
        Else => 'HOLD'
        """
        p_a, p_b = self.predict_proba(margins)
        decisions = np.full(p_a.shape, "HOLD", dtype="<U4")
        decisions[p_a >= prob_threshold] = "A"
        decisions[p_b >= prob_threshold] = "B"
        
        conf = np.abs(p_a - p_b)
        accepted = (decisions != "HOLD")
        
        return {
            "decisions": decisions,
            "accepted": accepted,
            "prob_a": p_a,
            "prob_b": p_b,
            "confidences": conf,
            "is_hold": (decisions == "HOLD"),
        }


class SelectiveRiskCoverageOptimizer:
    """
    Method 5: Selective Risk / Coverage Optimization (Geifman & El-Yaniv, 2017).
    
    Generates exact Risk vs. Coverage and Accuracy vs. Coverage profiles.
    Allows principled threshold selection for target coverage (e.g. 70%, 80%, 90%)
    or target selective risk on calibration folds.
    """
    @staticmethod
    def compute_curve(
        confidences: np.ndarray,
        predictions: np.ndarray,
        ground_truth: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Computes the complete empirical coverage-risk curve.
        """
        conf = np.asarray(confidences, dtype=np.float64)
        pred = np.asarray(predictions)
        gt = np.asarray(ground_truth)
        n = len(conf)
        
        if n == 0:
            return {
                "thresholds": np.array([]),
                "coverages": np.array([]),
                "risks": np.array([]),
                "accuracies": np.array([]),
                "aurc": 0.0,
                "e_aurc": 0.0,
            }
            
        # Sort descending by confidence
        sorted_indices = np.argsort(-conf)
        conf_sorted = conf[sorted_indices]
        pred_sorted = pred[sorted_indices]
        gt_sorted = gt[sorted_indices]
        
        correct_sorted = (pred_sorted == gt_sorted).astype(np.float64)
        
        # Unique thresholds to prevent duplicate threshold evaluations
        unique_thresholds, unique_indices = np.unique(conf_sorted[::-1], return_index=True)
        # Convert reverse unique indices back to descending order
        rev_idx = len(conf_sorted) - 1 - unique_indices
        rev_idx = np.sort(rev_idx)
        
        cum_correct = np.cumsum(correct_sorted)
        coverages = []
        risks = []
        accuracies = []
        thresholds_out = []
        
        for idx in rev_idx:
            n_acc = idx + 1
            cov = n_acc / n
            acc = cum_correct[idx] / n_acc
            risk = 1.0 - acc
            
            coverages.append(cov)
            risks.append(risk)
            accuracies.append(acc)
            thresholds_out.append(conf_sorted[idx])
            
        coverages = np.array(coverages, dtype=np.float64)
        risks = np.array(risks, dtype=np.float64)
        accuracies = np.array(accuracies, dtype=np.float64)
        thresholds_out = np.array(thresholds_out, dtype=np.float64)
        
        # Calculate AURC via trapezoidal rule over coverage
        sort_cov_idx = np.argsort(coverages)
        cov_asc = coverages[sort_cov_idx]
        risk_asc = risks[sort_cov_idx]
        
        if cov_asc[0] > 0.0:
            cov_asc = np.insert(cov_asc, 0, 0.0)
            risk_asc = np.insert(risk_asc, 0, risk_asc[0])
            
        aurc = float(_safe_trapezoid(risk_asc, cov_asc))
        
        # Optimal AURC: when predictions are perfectly ranked by confidence
        overall_acc = float(np.mean(correct_sorted))
        if overall_acc >= 1.0:
            optimal_aurc = 0.0
        elif overall_acc <= 0.0:
            optimal_aurc = 1.0
        else:
            cov_grid = np.linspace(0.0, 1.0, 1000)
            opt_risk = np.where(cov_grid <= overall_acc, 0.0, 1.0 - overall_acc / np.maximum(cov_grid, 1e-8))
            optimal_aurc = float(_safe_trapezoid(opt_risk, cov_grid))
            
        e_aurc = max(0.0, aurc - optimal_aurc)
        
        return {
            "thresholds": thresholds_out,
            "coverages": coverages,
            "risks": risks,
            "accuracies": accuracies,
            "aurc": aurc,
            "optimal_aurc": optimal_aurc,
            "e_aurc": e_aurc,
        }

    @staticmethod
    def select_threshold_for_coverage(
        calib_confidences: np.ndarray,
        target_coverage: float
    ) -> float:
        """
        Determines the threshold on calibration data that yields the specified target coverage.
        Target coverage: float in (0.0, 1.0] (e.g. 0.80 for 80% coverage).
        """
        if not (0.0 < target_coverage <= 1.0):
            raise ValueError(f"Target coverage must be in (0.0, 1.0], got {target_coverage}")
        conf = np.asarray(calib_confidences, dtype=np.float64)
        # Percentile corresponding to rejecting (1 - target_coverage) fraction
        q = (1.0 - target_coverage) * 100.0
        return float(np.percentile(conf, q))


class ConformalSelectiveGate:
    """
    Method 6: Conformal Selective Gate & Assumption Audit.
    
    Rigorous investigation of Split Conformal / Conformal Risk Control for streaming AAD.
    
    Assumption Audit:
      - Temporal Dependence: Consecutive sliding windows (e.g. 5s window with 0.5s hop)
        have 90% sample overlap, generating severe autoregressive dependence.
        Naive window-level exchangeability is VIOLATED.
      - Trial / Subject Structure: Trials and subjects introduce clustering shifts.
      - Exchangeable Units: When grouping non-overlapping windows or distinct trial units
        across calibration subjects, block-level exchangeability is satisfied.
        
    Implementation:
      - Fits non-conformity threshold on exchangeable calibration units.
      - Constructs prediction sets C(x) in {{A}, {B}, {A, B}, empty}.
      - Decision:
          C(x) = {A}       => 'A'
          C(x) = {B}       => 'B'
          C(x) = {A, B}    => 'HOLD' (uncertain, candidate ambiguity)
          C(x) = empty     => 'HOLD' (atypical outlier)
    """
    def __init__(self, error_rate_target: float = 0.10):
        if not (0.0 < error_rate_target < 1.0):
            raise ValueError(f"Error rate target must be in (0, 1), got {error_rate_target}")
        self.error_rate_target = float(error_rate_target)
        self.calibrated_quantile = 0.5
        self.assumption_audit = {
            "window_level_exchangeability": False,
            "reason": "Sliding windows with 0.5s step share 90% temporal overlap, violating i.i.d. exchangeability.",
            "valid_unit": "Trial-level or non-overlapping segment exchangeability across subjects.",
        }

    def fit_calibration_quantile(
        self,
        calib_probs_a: np.ndarray,
        calib_labels: np.ndarray
    ) -> float:
        """
        Fits split-conformal non-conformity quantile strictly on calibration set.
        calib_probs_a: posterior probability p(A)
        calib_labels: binary ground truth (1 for A, 0 for B)
        """
        p_a = np.asarray(calib_probs_a, dtype=np.float64)
        labels = np.asarray(calib_labels, dtype=np.float64)
        n = len(p_a)
        if n == 0:
            raise ValueError("Empty calibration data")
            
        # Non-conformity score: s_i = 1 - p(y_true | x_i)
        p_true = np.where(labels == 1.0, p_a, 1.0 - p_a)
        scores = 1.0 - p_true
        
        # Conformal quantile: ceil((n + 1) * (1 - alpha)) / n
        level = min(1.0, math.ceil((n + 1) * (1.0 - self.error_rate_target)) / n)
        self.calibrated_quantile = float(np.quantile(scores, level, method="higher"))
        return self.calibrated_quantile

    def predict_sets(self, probs_a: np.ndarray) -> Dict[str, Any]:
        """
        Evaluates conformal prediction sets and converts them to A/B/HOLD.
        """
        p_a = np.asarray(probs_a, dtype=np.float64)
        p_b = 1.0 - p_a
        
        # A candidate is in C(x) if 1 - p(y) <= calibrated_quantile => p(y) >= 1 - calibrated_quantile
        prob_cutoff = 1.0 - self.calibrated_quantile
        in_a = p_a >= prob_cutoff
        in_b = p_b >= prob_cutoff
        
        decisions = []
        set_sizes = []
        
        for ia, ib in zip(in_a, in_b):
            if ia and not ib:
                decisions.append("A")
                set_sizes.append(1)
            elif ib and not ia:
                decisions.append("B")
                set_sizes.append(1)
            elif ia and ib:
                # Both speakers plausible -> HOLD
                decisions.append("HOLD")
                set_sizes.append(2)
            else:
                # Neither speaker passed -> Outlier HOLD
                decisions.append("HOLD")
                set_sizes.append(0)
                
        decisions = np.array(decisions, dtype="<U4")
        accepted = (decisions != "HOLD")
        
        return {
            "decisions": decisions,
            "accepted": accepted,
            "set_sizes": np.array(set_sizes, dtype=np.int32),
            "is_hold": (decisions == "HOLD"),
            "prob_cutoff": prob_cutoff,
            "calibrated_quantile": self.calibrated_quantile,
        }
