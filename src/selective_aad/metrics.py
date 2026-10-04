import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

def _safe_trapezoid(y, x):
    """NumPy 2.0+ compatible trapezoidal integration."""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    elif hasattr(np, "trapz"):
        return np.trapz(y, x)
    else:
        from scipy import integrate
        return integrate.trapezoid(y, x)

def calculate_selective_metrics(
    decisions: np.ndarray,
    ground_truth: np.ndarray,
    label_a: Any = "A",
    label_b: Any = "B"
) -> Dict[str, float]:
    """
    Computes standard selective classification metrics given discrete decisions ('A', 'B', 'HOLD')
    and ground truth labels.
    """
    dec = np.asarray(decisions)
    gt = np.asarray(ground_truth)
    n_total = len(dec)
    
    if n_total == 0:
        return {
            "coverage": 0.0,
            "abstention_rate": 1.0,
            "selective_accuracy": 0.0,
            "selective_risk": 0.0,
            "raw_accuracy": 0.0,
            "accepted_count": 0,
            "rejected_count": 0,
            "total_count": 0,
        }
        
    accepted_mask = (dec != "HOLD")
    n_accepted = int(np.sum(accepted_mask))
    n_rejected = n_total - n_accepted
    coverage = n_accepted / n_total
    abstention_rate = 1.0 - coverage
    
    # Map gt labels if they are integers (e.g. 1 -> 'A', 0 -> 'B')
    if np.issubdtype(gt.dtype, np.number):
        gt_mapped = np.where(gt == 1, label_a, label_b)
    else:
        gt_mapped = gt
        
    if n_accepted > 0:
        correct_accepted = np.sum(dec[accepted_mask] == gt_mapped[accepted_mask])
        selective_accuracy = float(correct_accepted / n_accepted)
        selective_risk = 1.0 - selective_accuracy
    else:
        selective_accuracy = 0.0
        selective_risk = 0.0
        
    # Raw accuracy (forcing random choice or treating HOLD as error)
    correct_all = np.sum(dec == gt_mapped)
    raw_accuracy = float(correct_all / n_total)
    
    return {
        "coverage": float(coverage),
        "abstention_rate": float(abstention_rate),
        "selective_accuracy": float(selective_accuracy),
        "selective_risk": float(selective_risk),
        "raw_accuracy": float(raw_accuracy),
        "accepted_count": n_accepted,
        "rejected_count": n_rejected,
        "total_count": n_total,
    }


def compute_risk_coverage_curve(
    confidences: np.ndarray,
    predictions: np.ndarray,
    ground_truth: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Computes empirical coverage, selective risk, and selective accuracy across confidence thresholds.
    """
    conf = np.asarray(confidences, dtype=np.float64)
    pred = np.asarray(predictions)
    gt = np.asarray(ground_truth)
    
    # If gt is numeric 1/0 and pred is 'A'/'B', map
    if np.issubdtype(gt.dtype, np.number) and (pred.dtype.kind in ['U', 'S']):
        gt = np.where(gt == 1, "A", "B")
    elif (not np.issubdtype(gt.dtype, np.number)) and np.issubdtype(pred.dtype, np.number):
        gt = (gt == "A").astype(np.int64)
        
    n = len(conf)
    if n == 0:
        return {
            "coverages": np.array([0.0]),
            "risks": np.array([0.0]),
            "accuracies": np.array([0.0]),
            "thresholds": np.array([0.0]),
        }
        
    sort_idx = np.argsort(-conf)
    conf_sorted = conf[sort_idx]
    pred_sorted = pred[sort_idx]
    gt_sorted = gt[sort_idx]
    
    correct_sorted = (pred_sorted == gt_sorted).astype(np.float64)
    cum_correct = np.cumsum(correct_sorted)
    
    # Downsample points for computational efficiency if n is very large
    unique_thresholds, unique_indices = np.unique(conf_sorted[::-1], return_index=True)
    rev_idx = len(conf_sorted) - 1 - unique_indices
    rev_idx = np.sort(rev_idx)
    
    coverages = (rev_idx + 1) / n
    accuracies = cum_correct[rev_idx] / (rev_idx + 1)
    risks = 1.0 - accuracies
    thresholds = conf_sorted[rev_idx]
    
    return {
        "coverages": coverages,
        "risks": risks,
        "accuracies": accuracies,
        "thresholds": thresholds,
    }


def compute_aurc(coverages: np.ndarray, risks: np.ndarray) -> Tuple[float, float, float]:
    """
    Computes Area Under Risk-Coverage curve (AURC), Optimal AURC, and Excess AURC (E-AURC).
    """
    cov = np.asarray(coverages, dtype=np.float64)
    r = np.asarray(risks, dtype=np.float64)
    
    if len(cov) == 0:
        return 0.0, 0.0, 0.0
        
    sort_idx = np.argsort(cov)
    cov_sorted = cov[sort_idx]
    r_sorted = r[sort_idx]
    
    # Anchor to (0, r_0) and (1, r_final) if necessary
    if cov_sorted[0] > 0.0:
        cov_sorted = np.insert(cov_sorted, 0, 0.0)
        r_sorted = np.insert(r_sorted, 0, r_sorted[0])
    if cov_sorted[-1] < 1.0:
        cov_sorted = np.append(cov_sorted, 1.0)
        r_sorted = np.append(r_sorted, r_sorted[-1])
        
    aurc = float(_safe_trapezoid(r_sorted, cov_sorted))
    
    # Baseline accuracy at full coverage
    base_acc = 1.0 - float(r_sorted[-1])
    if base_acc >= 1.0:
        optimal_aurc = 0.0
    elif base_acc <= 0.0:
        optimal_aurc = 1.0
    else:
        # Theoretical optimal curve: zero error up to cov=base_acc, then (cov - base_acc)/cov
        grid = np.linspace(0.0, 1.0, 1000)
        opt_r = np.where(grid <= base_acc, 0.0, 1.0 - base_acc / np.maximum(grid, 1e-8))
        optimal_aurc = float(_safe_trapezoid(opt_r, grid))
        
    e_aurc = max(0.0, aurc - optimal_aurc)
    return aurc, optimal_aurc, e_aurc


def compute_ece(
    probs_a: np.ndarray,
    ground_truth_binary: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Any]:
    """
    Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
    with M equal-width confidence bins, along with reliability diagram bins.
    
    probs_a: array of p(A) in [0, 1]
    ground_truth_binary: 1 if A, 0 if B
    """
    probs = np.asarray(probs_a, dtype=np.float64)
    gt = np.asarray(ground_truth_binary, dtype=np.float64)
    n = len(probs)
    
    if n == 0:
        return {"ece": 0.0, "mce": 0.0, "bin_accs": [], "bin_confs": [], "bin_counts": []}
        
    # Binary confidence: max(p, 1-p)
    conf = np.maximum(probs, 1.0 - probs)
    # Accuracy: 1 if (p >= 0.5 and y==1) or (p < 0.5 and y==0)
    acc = ((probs >= 0.5) == (gt == 1.0)).astype(np.float64)
    
    bin_edges = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    
    bin_accs = []
    bin_confs = []
    bin_counts = []
    
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            in_bin = (conf >= low) & (conf <= high)
        else:
            in_bin = (conf >= low) & (conf < high)
            
        count = int(np.sum(in_bin))
        bin_counts.append(count)
        
        if count > 0:
            bin_acc = float(np.mean(acc[in_bin]))
            bin_conf = float(np.mean(conf[in_bin]))
            diff = abs(bin_acc - bin_conf)
            ece += (count / n) * diff
            mce = max(mce, diff)
            bin_accs.append(bin_acc)
            bin_confs.append(bin_conf)
        else:
            bin_accs.append(0.0)
            bin_confs.append((low + high) / 2.0)
            
    return {
        "ece": float(ece),
        "mce": float(mce),
        "bin_accs": bin_accs,
        "bin_confs": bin_confs,
        "bin_counts": bin_counts,
        "bin_edges": bin_edges.tolist(),
    }


def compute_brier_score(probs_a: np.ndarray, ground_truth_binary: np.ndarray) -> float:
    """Mean squared error of posterior probability estimate: (1/N) * sum((p_a - y)^2)."""
    p = np.asarray(probs_a, dtype=np.float64)
    y = np.asarray(ground_truth_binary, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def compute_nll(probs_a: np.ndarray, ground_truth_binary: np.ndarray, eps: float = 1e-12) -> float:
    """Negative log likelihood (binary cross entropy)."""
    p = np.clip(np.asarray(probs_a, dtype=np.float64), eps, 1.0 - eps)
    y = np.asarray(ground_truth_binary, dtype=np.float64)
    loss = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return float(np.mean(loss))


def compute_temporal_stability_metrics(
    decisions_sequence: np.ndarray,
    ground_truth_sequence: np.ndarray,
    step_sec: float = 0.5,
    label_a: str = "A",
    label_b: str = "B"
) -> Dict[str, float]:
    """
    Computes comprehensive temporal stability and hearing-aid steering metrics
    across a time-series of decisions ('A', 'B', 'HOLD').
    """
    dec = np.asarray(decisions_sequence)
    gt = np.asarray(ground_truth_sequence)
    n_steps = len(dec)
    
    if n_steps == 0:
        return {
            "total_duration_sec": 0.0,
            "switches_per_minute": 0.0,
            "false_switches_per_minute": 0.0,
            "time_to_first_lock_sec": 0.0,
            "time_to_correct_switch_sec": 0.0,
            "mean_incorrect_lock_duration_sec": 0.0,
            "mean_hold_duration_sec": 0.0,
            "pct_correctly_locked": 0.0,
            "pct_incorrectly_locked": 0.0,
            "pct_hold": 100.0,
        }
        
    duration_sec = n_steps * step_sec
    duration_min = duration_sec / 60.0
    
    if np.issubdtype(gt.dtype, np.number):
        gt_mapped = np.where(gt == 1, label_a, label_b)
    else:
        gt_mapped = gt
        
    # State tracking
    switches = 0
    false_switches = 0
    time_to_first_lock = None
    
    correct_lock_steps = 0
    incorrect_lock_steps = 0
    hold_steps = 0
    
    # Durations of runs
    incorrect_run_lengths = []
    hold_run_lengths = []
    
    current_incorrect_run = 0
    current_hold_run = 0
    
    prev_state = None
    
    for i in range(n_steps):
        d = dec[i]
        target = gt_mapped[i]
        
        # State distribution
        if d == "HOLD":
            hold_steps += 1
            current_hold_run += 1
            if current_incorrect_run > 0:
                incorrect_run_lengths.append(current_incorrect_run)
                current_incorrect_run = 0
        elif d == target:
            correct_lock_steps += 1
            if current_incorrect_run > 0:
                incorrect_run_lengths.append(current_incorrect_run)
                current_incorrect_run = 0
            if current_hold_run > 0:
                hold_run_lengths.append(current_hold_run)
                current_hold_run = 0
        else:
            # Incorrect lock
            incorrect_lock_steps += 1
            current_incorrect_run += 1
            if current_hold_run > 0:
                hold_run_lengths.append(current_hold_run)
                current_hold_run = 0
                
        # First lock latency
        if time_to_first_lock is None and d in [label_a, label_b]:
            time_to_first_lock = i * step_sec
            
        # Switch detection: transition between actual decision states or from HOLD into a state
        if prev_state is not None and d != prev_state and d in [label_a, label_b]:
            switches += 1
            if d != target:
                false_switches += 1
                
        prev_state = d
        
    # Flush remaining runs
    if current_incorrect_run > 0:
        incorrect_run_lengths.append(current_incorrect_run)
    if current_hold_run > 0:
        hold_run_lengths.append(current_hold_run)
        
    switches_per_min = switches / max(duration_min, 1e-4)
    false_switches_per_min = false_switches / max(duration_min, 1e-4)
    
    mean_incorrect_duration = (
        float(np.mean(incorrect_run_lengths) * step_sec) if incorrect_run_lengths else 0.0
    )
    mean_hold_duration = (
        float(np.mean(hold_run_lengths) * step_sec) if hold_run_lengths else 0.0
    )
    
    time_first_lock_sec = time_to_first_lock if time_to_first_lock is not None else duration_sec
    
    pct_correct = (correct_lock_steps / n_steps) * 100.0
    pct_incorrect = (incorrect_lock_steps / n_steps) * 100.0
    pct_hold = (hold_steps / n_steps) * 100.0
    
    return {
        "total_duration_sec": float(duration_sec),
        "switches_per_minute": float(switches_per_min),
        "false_switches_per_minute": float(false_switches_per_min),
        "time_to_first_lock_sec": float(time_first_lock_sec),
        "mean_incorrect_lock_duration_sec": float(mean_incorrect_duration),
        "mean_hold_duration_sec": float(mean_hold_duration),
        "pct_correctly_locked": float(pct_correct),
        "pct_incorrectly_locked": float(pct_incorrect),
        "pct_hold": float(pct_hold),
    }
