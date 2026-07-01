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

    def predict_window(self, margin, pearson_a=None, pearson_b=None, use_pearson=False):
        """
        Given window-level statistics, returns the prediction and whether it was accepted.
        
        Args:
            margin (float): Confidence margin (e.g., from Conformer's confidence head).
            pearson_a (float): Pearson correlation for Stream A.
            pearson_b (float): Pearson correlation for Stream B.
            use_pearson (bool): If True, confidence is derived from abs(pearson_a - pearson_b).
                                If False, confidence is derived directly from the margin.
                                
        Returns:
            dict: {
                "prediction": 0 for Stream A, 1 for Stream B.
                "confidence": The normalized confidence score [0, 1].
                "accepted": True if confidence >= threshold, False otherwise.
                "margin": Raw margin for downstream analysis.
                "pearson_diff": Raw pearson difference for downstream analysis.
            }
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
            
        # Compute normalized confidence [0, 1]
        # Margin is in [-1, 1]. Confidence = abs(margin) in [0, 1].
        if use_pearson:
            # Pearson diff is in [0, 2], typically much smaller. 
            # We scale it arbitrarily or just use the absolute difference.
            # To treat it as a probability-like score, we could pass it through a sigmoid or just use it raw.
            # Since we sweep thresholds, raw is fine, but bounded is better.
            confidence = abs(pearson_a - pearson_b)
        else:
            confidence = abs(margin)
            
        # Decision
        accepted = confidence >= self.threshold
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "accepted": accepted,
            "margin": margin,
            "pearson_a": pearson_a,
            "pearson_b": pearson_b,
            "pearson_diff": abs(pearson_a - pearson_b) if (pearson_a is not None and pearson_b is not None) else None
        }

    def predict_trial(self, window_results, aggregation="majority"):
        """
        Aggregates window-level results into a trial-level selective prediction.
        
        Args:
            window_results (list of dict): The output of predict_window for all windows in a trial.
            aggregation (str): Strategy to aggregate ('majority', 'weighted_majority', 'accumulated_pearson')
            
        Returns:
            dict: Trial-level prediction, confidence, and acceptance.
        """
        if not window_results:
            return {
                "prediction": None, 
                "confidence": 0.0, 
                "accepted": False,
                "accepted_windows_count": 0,
                "total_windows_count": 0
            }
            
        accepted_windows = [w for w in window_results if w["accepted"]]
        
        # Strategy 1: Reject trial if NO windows are accepted
        if len(accepted_windows) == 0:
            return {
                "prediction": -1, 
                "confidence": 0.0, 
                "accepted": False,
                "reason": "All windows rejected",
                "accepted_windows_count": 0,
                "total_windows_count": len(window_results)
            }
            
        if aggregation == "majority":
            # Majority vote over ACCEPTED windows
            preds = [w["prediction"] for w in accepted_windows]
            count_1 = sum(preds)
            count_0 = len(preds) - count_1
            
            trial_pred = 1 if count_1 > count_0 else 0
            
            # Trial confidence = margin of the vote (e.g., 5-3 vote -> 2/8 = 0.25)
            # Or average confidence of accepted windows.
            trial_conf = abs(count_1 - count_0) / len(preds)
            
            return {
                "prediction": trial_pred,
                "confidence": trial_conf,
                "accepted": True,
                "accepted_windows_count": len(accepted_windows),
                "total_windows_count": len(window_results)
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
                "accepted_windows_count": len(accepted_windows),
                "total_windows_count": len(window_results)
            }
            
        return {"prediction": None, "confidence": 0.0, "accepted": False}
