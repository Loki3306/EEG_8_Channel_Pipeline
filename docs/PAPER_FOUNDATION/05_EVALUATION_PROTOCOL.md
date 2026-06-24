# 05 Evaluation Protocol

## Base Accuracy Metrics
1. **Window Accuracy**: The percentage of 3-second windows where the model correctly identifies the attended stream (`sim_a > sim_b`).
2. **Trial Accuracy**: The percentage of full trials where the average margin across all windows in the trial is greater than 0.

## Confidence & Calibration Metrics
To evaluate the secondary XGBoost confidence model, the following metrics were formally adopted:

1. **AUROC (Area Under the Receiver Operating Characteristic Curve)**:
   - Measures the ability of the confidence score to discriminate between correctly and incorrectly predicted windows.
   - Values > 0.5 indicate predictive power; typical production values hit ~0.75-0.80.

2. **Brier Score**:
   - Measures the mean squared difference between the predicted probability (confidence) and the actual outcome (1 for correct, 0 for incorrect).
   - Lower is better. Measures calibration.

3. **ECE (Expected Calibration Error)**:
   - Bins predictions by confidence and measures the absolute difference between average confidence and true accuracy in each bin.
   - Lower is better. A perfectly calibrated model has an ECE of 0.

4. **E-AURC (Empirical Area Under the Risk-Coverage Curve)**:
   - Measures the integral of the Risk (Error Rate) as Coverage (fraction of accepted windows) varies from 0 to 1.
   - Lower is better. Useful for comparing selective prediction systems.

## Selective Prediction Metrics
1. **Coverage**: The percentage of total windows that the system chooses to accept (not reject).
   - e.g., A Coverage of 80% means the bottom 20% most uncertain windows were discarded.
2. **Selective Accuracy**: The Window Accuracy calculated *only* on the accepted windows.
   - The primary goal of the runtime system is to demonstrate monotonically increasing Selective Accuracy as Coverage decreases.
