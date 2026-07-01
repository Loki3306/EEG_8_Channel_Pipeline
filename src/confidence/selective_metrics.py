import numpy as np

def calculate_selective_risk(y_true, y_pred, y_conf, threshold):
    """
    Calculates metrics for selective classification at a specific threshold.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_conf: Confidence scores
        threshold: Confidence threshold for acceptance
        
    Returns:
        dict: Coverage, Accepted Accuracy, Rejected Accuracy, Selective Risk
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_conf = np.array(y_conf)
    
    accepted_mask = y_conf >= threshold
    rejected_mask = ~accepted_mask
    
    coverage = np.mean(accepted_mask)
    
    if np.sum(accepted_mask) > 0:
        accepted_acc = np.mean(y_true[accepted_mask] == y_pred[accepted_mask])
        selective_risk = 1.0 - accepted_acc
    else:
        accepted_acc = 0.0
        selective_risk = 0.0
        
    if np.sum(rejected_mask) > 0:
        rejected_acc = np.mean(y_true[rejected_mask] == y_pred[rejected_mask])
    else:
        rejected_acc = 0.0
        
    return {
        "coverage": float(coverage),
        "accepted_accuracy": float(accepted_acc),
        "rejected_accuracy": float(rejected_acc),
        "selective_risk": float(selective_risk),
        "accepted_count": int(np.sum(accepted_mask)),
        "rejected_count": int(np.sum(rejected_mask))
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
