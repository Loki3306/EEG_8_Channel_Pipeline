# 15 Reproducibility Guide

## Environment Setup
1. **Repository**: `github.com/Loki3306/EEG_8_Channel_Pipeline` (Branch: `phase-4`).
2. **Environment**: Python 3.10+, PyTorch 2.0+, XGBoost, Scikit-learn, SciPy.
3. **Data Mounting**:
   - DTU Preprocessed Tensors (`/kaggle/input/...`)
   - DTU Gammatone Envelopes (`gammatone_envelopes.pkl`)
   - KUL Raw Dataset (`/kaggle/input/datasets/lowk1ee/s1-klu`)
   - KUL Audio Dataset (`/kaggle/input/datasets/lowk1ee/audio-klu`)

## Reproducing Baseline Training
Run the standard LOSO training to generate the 18 folds:
```bash
python training/train_matchnet_loso.py
```
This generates `checkpoints/matchnet_fold_S<X>_best.pth`.

## Reproducing Confidence Engineering
1. **Generate Predictions**:
```bash
python training/export_subject_distance.py
```
This evaluates all LOSO folds and dumps `subject_distance_predictions.csv`.

2. **Train Confidence Model**:
```bash
python analysis/step_5_0a_train_final_model.py
```
This trains the XGBoost runtime and saves `models/confidence_model.json`.

## Reproducing Confidence Audits
Run the behavior audit to generate Reliability and Selective Accuracy curves:
```bash
python analysis/step_5_1_behavior_audit.py
```

## Reproducing KUL Transfer
To run the full 28-band preprocessing and cross-dataset evaluation:
```bash
python analysis/step_6_8_kul_ablation_and_confidence.py
```
This will print out the window accuracy and confidence AUROC for multiple window lengths on KUL S1.
