import numpy as np

class SelectivePredictor:
    """
    Unified Confidence Engine for Selective AAD.
    Consolidates predictions and confidence proxies into a single source of truth.
    """
    def __init__(self, threshold=0.5):
        """
        Initialize the predictor with a given abstention threshold.
        threshold: float in [0.0, 1.0]. 
                   If confidence < threshold, the prediction is rejected.
        """
        self.threshold = threshold

    def predict_window(self, margin, pearson_a=None, pearson_b=None, use_pearson=False, learned_confidence=None):
        """
        Predicts whether a single window should be accepted or rejected.
        
        Args:
            margin (float): The margin between Stream A and Stream B metrics.
            pearson_a (float, optional): Pearson correlation for Stream A.
            pearson_b (float, optional): Pearson correlation for Stream B.
            use_pearson (bool): Whether to use explicit Pearson correlations.
            learned_confidence (float, optional): The probability score from the model's confidence head.
            
        Returns:
            dict: Window prediction details including:
                "prediction": 1 for Stream B, 0 for Stream A.
                "confidence": The normalized confidence score [0, 1].
                "accepted": True if confidence >= threshold, False otherwise.
                "margin": Raw margin for downstream analysis.
                "pearson_diff": Raw pearson difference for downstream analysis.
        """
        # Determine raw prediction
        # Margin > 0 implies Stream B (since margin = P(B) - P(A))
        # Wait, usually margin = P(1) - P(0) or similar.
        # Let's standardize: 
        # If margin > 0, prediction = 1 (Stream B). Else 0 (Stream A).
        prediction = 1 if margin > 0 else 0
        
        if use_pearson and pearson_a is not None and pearson_b is not None:
            # Re-evaluate prediction based on Pearson if required
            prediction = 0 if pearson_a > pearson_b else 1
            
        # Calculate window confidence
        if learned_confidence is not None:
            # Use the actual output from the model's confidence head
            confidence = float(learned_confidence)
        elif use_pearson:
            # Fallback: Confidence is defined as the absolute Pearson correlation margin
            confidence = abs(pearson_a - pearson_b) if (pearson_a is not None and pearson_b is not None) else 0.0
        else:
            # Placeholder for latent-based confidence
            confidence = abs(margin)
            
        accepted = bool(confidence >= self.threshold)
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "accepted": accepted,
            "margin": margin,
            "pearson_a": pearson_a,
            "pearson_b": pearson_b,
            "pearson_diff": abs(pearson_a - pearson_b) if (pearson_a is not None and pearson_b is not None) else None
        }

    def predict_trial(self, window_results, aggregation="majority", min_accept_ratio=0.0):
        """
        Aggregates window-level results into a trial-level selective prediction.
        
        Args:
            window_results (list of dict): The output of predict_window for all windows in a trial.
            aggregation (str): Strategy to aggregate ('majority', 'weighted_majority', 'accumulated_pearson')
            min_accept_ratio (float): Minimum ratio of accepted windows required to accept the trial.
            
        Returns:
            dict: Trial-level prediction, confidence, and acceptance.
        """
        if not window_results:
            return {
                "prediction": None, 
                "confidence": 0.0, 
                "accepted": False,
                "reason": "No windows provided",
                "accepted_windows_count": 0,
                "total_windows_count": 0,
                "mean_window_confidence": 0.0,
                "median_window_confidence": 0.0
            }
            
        accepted_windows = [w for w in window_results if w["accepted"]]
        total_windows = len(window_results)
        
        all_confs = [w["confidence"] for w in window_results]
        mean_conf = sum(all_confs) / total_windows
        median_conf = sorted(all_confs)[total_windows // 2]
        
        accept_ratio = len(accepted_windows) / total_windows
        
        # Strategy 1: Reject trial if accept ratio is too low
        if accept_ratio <= min_accept_ratio or len(accepted_windows) == 0:
            return {
                "prediction": -1, 
                "confidence": mean_conf, 
                "accepted": False,
                "reason": f"Rejected: {len(accepted_windows)}/{total_windows} accepted windows <= {min_accept_ratio*100}% threshold",
                "accepted_windows_count": len(accepted_windows),
                "total_windows_count": total_windows,
                "mean_window_confidence": mean_conf,
                "median_window_confidence": median_conf
            }
            
        if aggregation == "majority":
            # Majority vote over ACCEPTED windows
            preds = [w["prediction"] for w in accepted_windows]
            count_1 = sum(preds)
            count_0 = len(preds) - count_1
            
            trial_pred = 1 if count_1 > count_0 else 0
            
            # Trial confidence = average confidence of accepted windows
            trial_conf = sum([w["confidence"] for w in accepted_windows]) / len(accepted_windows)
            
            return {
                "prediction": trial_pred,
                "confidence": trial_conf,
                "accepted": True,
                "reason": f"Accepted: {len(accepted_windows)}/{total_windows} > {min_accept_ratio*100}% threshold",
                "accepted_windows_count": len(accepted_windows),
                "total_windows_count": total_windows,
                "mean_window_confidence": mean_conf,
                "median_window_confidence": median_conf
            }
            
        elif aggregation == "accumulated_pearson":
            # Sum pearsons over ACCEPTED windows
            sum_a = sum([w["pearson_a"] for w in accepted_windows if w["pearson_a"] is not None])
            sum_b = sum([w["pearson_b"] for w in accepted_windows if w["pearson_b"] is not None])
            
            trial_pred = 0 if sum_a > sum_b else 1
            # Confidence proxy is the absolute difference in sums normalized by window count
            trial_conf = abs(sum_a - sum_b) / max(len(accepted_windows), 1)
            
            return {
                "prediction": trial_pred,
                "confidence": trial_conf,
                "accepted": True,
                "reason": f"Accepted: {len(accepted_windows)}/{total_windows} > {min_accept_ratio*100}% threshold",
                "accepted_windows_count": len(accepted_windows),
                "total_windows_count": total_windows,
                "mean_window_confidence": mean_conf,
                "median_window_confidence": median_conf
            }
            
        return {"prediction": None, "confidence": 0.0, "accepted": False}
