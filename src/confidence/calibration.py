import numpy as np
from sklearn.metrics import brier_score_loss

def calculate_ece(y_true, y_prob, n_bins=10):
    """
    Calculates the Expected Calibration Error (ECE).
    """
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    
    ece = 0.0
    for i in range(n_bins):
        bin_idx = (binids == i)
        if np.sum(bin_idx) > 0:
            bin_acc = np.mean(y_true[bin_idx])
            bin_conf = np.mean(y_prob[bin_idx])
            prob = np.sum(bin_idx) / len(y_prob)
            ece += prob * np.abs(bin_acc - bin_conf)
            
    return ece

def calculate_mce(y_true, y_prob, n_bins=10):
    """
    Calculates the Maximum Calibration Error (MCE).
    """
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    
    mce = 0.0
    for i in range(n_bins):
        bin_idx = (binids == i)
        if np.sum(bin_idx) > 0:
            bin_acc = np.mean(y_true[bin_idx])
            bin_conf = np.mean(y_prob[bin_idx])
            mce = max(mce, np.abs(bin_acc - bin_conf))
            
    return mce

def calculate_brier_score(y_true, y_prob):
    """
    Calculates the Brier Score.
    """
    return brier_score_loss(y_true, y_prob)

def get_calibration_curve(y_true, y_prob, n_bins=10):
    """
    Returns true fraction of positives and mean predicted probability per bin.
    """
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    
    prob_true = []
    prob_pred = []
    
    for i in range(n_bins):
        bin_idx = (binids == i)
        if np.sum(bin_idx) > 0:
            prob_true.append(np.mean(y_true[bin_idx]))
            prob_pred.append(np.mean(y_prob[bin_idx]))
            
    return np.array(prob_true), np.array(prob_pred)
