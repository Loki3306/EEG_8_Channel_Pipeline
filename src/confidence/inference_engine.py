import numpy as np
import xgboost as xgb
import os

class ConfidenceState:
    def __init__(self):
        self.margins = []
        self.predictions = []

    def update(self, margin, prediction):
        self.margins.append(margin)
        self.predictions.append(prediction)

def build_confidence_features(margin, sim_a, sim_b, current_prediction, state: ConfidenceState):
    # 1. Similarities
    sim_chosen = max(sim_a, sim_b)
    sim_unchosen = min(sim_a, sim_b)
    
    # 2. Rolling Std (must include current margin, up to 5 elements)
    hist_margins = state.margins[-4:] + [margin]
    if len(hist_margins) < 2:
        rolling_std = 0.0
    else:
        # ddof=1 matches pandas default
        rolling_std = np.std(hist_margins, ddof=1)
        
    # 3. Trial Consistency (history must exclude current prediction)
    if len(state.predictions) == 0:
        consistency = 1.0
    else:
        history_arr = np.array(state.predictions)
        consistency = np.mean(history_arr == current_prediction)
        
    return [float(margin), float(sim_chosen), float(sim_unchosen), float(rolling_std), float(consistency)]

class ConfidenceEngine:
    def __init__(self, model_path, threshold=0.80):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        self.threshold = threshold
        self.state = ConfidenceState()
        
    def predict_with_confidence(self, eeg_window, sim_a, sim_b):
        """
        Takes similarities, computes margin and prediction, returns confidence decision.
        """
        margin = sim_a - sim_b
        prediction = 1 if margin >= 0 else 0
        
        features = build_confidence_features(margin, sim_a, sim_b, prediction, self.state)
        
        # XGBoost expects 2D array
        features_2d = np.array([features])
        confidence = self.model.predict_proba(features_2d)[0, 1]
        
        accept = bool(confidence >= self.threshold)
        
        # Update state for NEXT window
        self.state.update(margin, prediction)
        
        return {
            "prediction": prediction,
            "confidence": float(confidence),
            "accept": accept,
            "margin": float(margin),
            "features_used": features
        }
    
    def reset_trial(self):
        self.state = ConfidenceState()
