import numpy as np
from sklearn.metrics import roc_curve, auc

def calculate_selective_risk(y_true, y_pred, y_conf, threshold):
    """
    Calculates Selective Risk and related metrics for a given confidence threshold.
    
    Args:
        y_true (list): Ground truth labels
        y_pred (list): Predicted labels
        y_conf (list): Confidence scores
        threshold (float): Confidence threshold for acceptance
        
    Returns:
        dict: Metrics dictionary
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_conf = np.array(y_conf)
    
    accepted_mask = y_conf >= threshold
    rejected_mask = ~accepted_mask
    
    accepted_count = int(np.sum(accepted_mask))
    rejected_count = int(np.sum(rejected_mask))
    total_count = len(y_true)
    
    coverage = accepted_count / max(1, total_count)
    
    if accepted_count > 0:
        accepted_correct = np.sum(y_true[accepted_mask] == y_pred[accepted_mask])
        accepted_accuracy = accepted_correct / accepted_count
    else:
        accepted_accuracy = 0.0
        
    if rejected_count > 0:
        rejected_correct = np.sum(y_true[rejected_mask] == y_pred[rejected_mask])
        rejected_accuracy = rejected_correct / rejected_count
    else:
        rejected_accuracy = 0.0
        
    overall_correct = np.sum(y_true == y_pred)
    overall_accuracy = overall_correct / max(1, total_count)
        
    # Selective Risk is explicitly defined as the error rate of ACCEPTED predictions
    selective_risk = 1.0 - accepted_accuracy
    
    return {
        "coverage": float(coverage),
        "overall_accuracy": float(overall_accuracy),
        "accepted_accuracy": float(accepted_accuracy),
        "rejected_accuracy": float(rejected_accuracy),
        "selective_risk": float(selective_risk),
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "total_count": total_count
    }

def get_risk_coverage_curve(y_true, y_pred, y_conf):
    """
    Computes the Risk-Coverage curve across all unique confidence thresholds.
    """
    thresholds = np.sort(np.unique(y_conf))[::-1]
    
    coverages = []
    risks = []
    accuracies = []
    
    for t in thresholds:
        metrics = calculate_selective_risk(y_true, y_pred, y_conf, t)
        coverages.append(metrics["coverage"])
        risks.append(metrics["selective_risk"])
        accuracies.append(metrics["accepted_accuracy"])
        
    return np.array(thresholds), np.array(coverages), np.array(risks), np.array(accuracies)

def calculate_aurc(coverages, risks):
    """
    Calculates Area Under the Risk-Coverage Curve (AURC).
    Sorts coverages (ascending) for proper trapezoidal integration.
    """
    sort_idx = np.argsort(coverages)
    cov_sorted = np.array(coverages)[sort_idx]
    risk_sorted = np.array(risks)[sort_idx]
    
    # Calculate area using trapezoidal rule
    aurc = np.trapz(risk_sorted, cov_sorted)
    return float(aurc)
