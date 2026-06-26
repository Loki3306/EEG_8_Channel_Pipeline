import numpy as np

def compute_covariance_matrix(X):
    """
    Computes the spatial covariance matrix of an EEG window.
    X: shape (channels, samples)
    Returns: shape (channels, channels)
    """
    # X must be zero mean along time axis
    X_centered = X - np.mean(X, axis=1, keepdims=True)
    cov = (X_centered @ X_centered.T) / (X.shape[1] - 1)
    return cov

def fractional_matrix_power(M, power, reg=1e-6):
    """
    Computes M^power using eigenvalue decomposition.
    Adds regularization for numerical stability.
    """
    M_reg = M + np.eye(M.shape[0]) * reg
    w, v = np.linalg.eigh(M_reg)
    w = np.maximum(w, 1e-12)
    return v @ np.diag(w ** power) @ v.T

def compute_reference_covariance(windows):
    """
    Computes the mean covariance matrix across a list of windows.
    windows: list of arrays of shape (channels, samples)
    """
    covs = [compute_covariance_matrix(w) for w in windows]
    return np.mean(covs, axis=0)

def apply_alignment(X, R):
    """
    Applies the alignment matrix R to the EEG window X.
    X: shape (channels, samples)
    R: shape (channels, channels)
    """
    return R @ X

def prepare_alignment_matrices(source_windows, target_windows=None):
    """
    Computes alignment matrices based on the mode.
    If target_windows is None, it just computes the whitening matrix for source_windows (Standard EA).
    If target_windows is provided, it computes the whitening matrix for source AND recoloring matrix to target.
    
    Returns: 
    - R_whiten: matrix to whiten source_windows
    - R_recolor: matrix to recolor source_windows to match target_windows (or Identity if None)
    """
    cov_source = compute_reference_covariance(source_windows)
    R_whiten = fractional_matrix_power(cov_source, -0.5)
    
    if target_windows is not None:
        cov_target = compute_reference_covariance(target_windows)
        R_recolor = fractional_matrix_power(cov_target, 0.5)
    else:
        R_recolor = np.eye(cov_source.shape[0])
        
    return R_whiten, R_recolor
