# 14 Limitations

## 1. DTU-Only Training
The primary MatchNet architecture and the XGBoost confidence model were trained exclusively on the DTU dataset. While we have proven zero-shot transfer to KUL Subject 1 using identical preprocessing, full generalized transfer across all KUL subjects remains untested.

## 2. Lack of Real-Time Switching Datasets
Both DTU and KUL enforce strict, continuous attention on a single speaker per trial. Real-world auditory environments involve rapid switching of attention. The confidence framework can theoretically handle switching by using `trial_consistency` drops as a trigger to reset tracking, but this has not been evaluated on empirical switching datasets.

## 3. 28-Band Gammatone Dependency
The model is extremely fragile to acoustic representation shifts. We discovered that MatchNet relies entirely on the precise geometry of the 28-band Gammatone envelopes compressed by `^0.3`. Any pipeline deployed in real-time must exactly replicate this complex MATLAB-originating filterbank, introducing computational overhead.

## 4. Suboptimal Baseline Architecture
While MatchNet outperforms Ridge Regression, modern architectures (like VLAAI or ATCNet) demonstrate superior base accuracy. However, those architectures were bypassed in this project because their spatial filtering mechanisms exhibited catastrophic data leakage during early testing. 

## 5. Confidence Bound by MatchNet Info
The Confidence model is a downstream consumer of MatchNet's latent outputs. If MatchNet outputs high similarities for an artifact (a "confident error"), the XGBoost model has no raw EEG access to correct it. It relies entirely on temporal instability (`rolling_std_margin`) to catch these errors.
