import os
import sys
import numpy as np
import copy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.kul_cached_dataset import KULCachedLoader
from baselines.ridge_aad import TrialExample, lagged_eeg_matrix, feature_statistics, standardize_features
from training.train_kul_ridge_loso import evaluate_trial_majority_vote

FS = 64
RIDGE_LAMBDA = 100.0
LAGS = 32
LAG_STEP_MS = 16
WINDOW_SEC = 10
SUBJECTS_TO_EVAL = ["S1", "S5", "S11", "S13"]

def get_prediction(example, weights, feature_mean, feature_std):
    x = lagged_eeg_matrix(example.eeg, lags=LAGS, lag_ms=None, lag_step_ms=LAG_STEP_MS)
    x = standardize_features(x, feature_mean, feature_std)
    pred = x @ weights
    pred = pred - pred.mean()
    pred = pred / (pred.std() + 1e-12)
    return pred

def run_sanity_checks():
    print("Loading KUL Cache...")
    loader = KULCachedLoader(REPO_ROOT / "data" / "processed_kul")
    all_subject_data = loader.load_all()
    
    subject_paths = sorted(all_subject_data.keys())
    
    # Pre-format KUL dicts into TrialExamples
    subject_examples = {}
    for sub_id in subject_paths:
        examples = []
        for t in all_subject_data[sub_id]:
            ex = TrialExample(
                subject=sub_id,
                trial_index=t["meta"].get("TrialID", 0),
                eeg=t["eeg"].numpy().T,
                wav_a=t["audio_a"].numpy().mean(axis=0),
                wav_b=t["audio_b"].numpy().mean(axis=0),
                label=1
            )
            examples.append(ex)
        subject_examples[sub_id] = examples
        
    results = {sub: {} for sub in SUBJECTS_TO_EVAL}
    
    for held_out in SUBJECTS_TO_EVAL:
        if held_out not in subject_paths:
            print(f"Skipping {held_out}, not in cache.")
            continue
            
        print(f"\n========================================")
        print(f"Sanity Check Fold: {held_out}")
        print(f"========================================")
        
        fold_train_examples = []
        for other_id in subject_paths:
            if other_id != held_out:
                fold_train_examples.extend(subject_examples[other_id])
                
        print("1. Computing feature statistics...")
        feature_mean, feature_std = feature_statistics(
            fold_train_examples, lags=LAGS, lag_ms=None, lag_step_ms=LAG_STEP_MS, channel_ids=None
        )
        
        feature_count = feature_mean.shape[0]
        base_xtx = np.zeros((feature_count, feature_count), dtype=float)
        base_xty = np.zeros(feature_count, dtype=float)
        
        rand_xtx = np.zeros((feature_count, feature_count), dtype=float)
        rand_xty = np.zeros(feature_count, dtype=float)
        
        print("2. Accumulating Ridge Matrices (Baseline & Random Label)...")
        for example in fold_train_examples:
            x = lagged_eeg_matrix(example.eeg, lags=LAGS, lag_ms=None, lag_step_ms=LAG_STEP_MS)
            x = standardize_features(x, feature_mean, feature_std)
            xtx = x.T @ x
            
            # Baseline
            base_xtx += xtx
            base_xty += x.T @ example.wav_a
            
            # Random Label Training
            rand_xtx += xtx
            if np.random.rand() > 0.5:
                rand_xty += x.T @ example.wav_a
            else:
                rand_xty += x.T @ example.wav_b
                
        print("3. Solving Ridge...")
        base_weights = np.linalg.solve(base_xtx + RIDGE_LAMBDA * np.eye(feature_count), base_xty)
        rand_weights = np.linalg.solve(rand_xtx + RIDGE_LAMBDA * np.eye(feature_count), rand_xty)
        
        print("4. Evaluating Sanity Conditions...")
        test_examples = subject_examples[held_out]
        
        cond_correct = {
            "Baseline": 0,
            "Zero EEG": 0,
            "Random EEG": 0,
            "Shuffle EEG": 0,
            "Shuffle Audio": 0,
            "Circular Shift": 0,
            "Swap Labels": 0,
            "Random Train": 0,
        }
        
        num_trials = len(test_examples)
        
        for i, example in enumerate(test_examples):
            # Baseline Pred
            pred_base = get_prediction(example, base_weights, feature_mean, feature_std)
            
            # Zero EEG
            ex_zero = copy.deepcopy(example)
            ex_zero = TrialExample(ex_zero.subject, ex_zero.trial_index, np.zeros_like(ex_zero.eeg), ex_zero.wav_a, ex_zero.wav_b, ex_zero.label)
            pred_zero = get_prediction(ex_zero, base_weights, feature_mean, feature_std)
            
            # Random EEG
            ex_rand = copy.deepcopy(example)
            ex_rand = TrialExample(ex_rand.subject, ex_rand.trial_index, np.random.randn(*ex_rand.eeg.shape), ex_rand.wav_a, ex_rand.wav_b, ex_rand.label)
            pred_rand = get_prediction(ex_rand, base_weights, feature_mean, feature_std)
            
            # Shuffle EEG (Time)
            shuffled_eeg = np.copy(example.eeg)
            for ch in range(shuffled_eeg.shape[1]):
                np.random.shuffle(shuffled_eeg[:, ch])
            ex_shuff = copy.deepcopy(example)
            ex_shuff = TrialExample(ex_shuff.subject, ex_shuff.trial_index, shuffled_eeg, ex_shuff.wav_a, ex_shuff.wav_b, ex_shuff.label)
            pred_shuff = get_prediction(ex_shuff, base_weights, feature_mean, feature_std)
            
            # Random Train Pred
            pred_rtrain = get_prediction(example, rand_weights, feature_mean, feature_std)
            
            # Evals
            # 1. Baseline
            if evaluate_trial_majority_vote(pred_base, example.wav_a, example.wav_b, WINDOW_SEC)[0]: cond_correct["Baseline"] += 1
            # 2. Zero EEG
            if evaluate_trial_majority_vote(pred_zero, example.wav_a, example.wav_b, WINDOW_SEC)[0]: cond_correct["Zero EEG"] += 1
            # 3. Random EEG
            if evaluate_trial_majority_vote(pred_rand, example.wav_a, example.wav_b, WINDOW_SEC)[0]: cond_correct["Random EEG"] += 1
            # 4. Shuffle EEG
            if evaluate_trial_majority_vote(pred_shuff, example.wav_a, example.wav_b, WINDOW_SEC)[0]: cond_correct["Shuffle EEG"] += 1
            
            # 5. Shuffle Audio
            other_idx = np.random.choice([j for j in range(num_trials) if j != i]) if num_trials > 1 else i
            other_ex = test_examples[other_idx]
            if evaluate_trial_majority_vote(pred_base, other_ex.wav_a, other_ex.wav_b, WINDOW_SEC)[0]: cond_correct["Shuffle Audio"] += 1
            
            # 6. Circular Shift
            shift_samples = 5 * FS
            shift_a = np.roll(example.wav_a, shift_samples)
            shift_b = np.roll(example.wav_b, shift_samples)
            if evaluate_trial_majority_vote(pred_base, shift_a, shift_b, WINDOW_SEC)[0]: cond_correct["Circular Shift"] += 1
            
            # 7. Swap Labels
            # The model predicts A vs B based on correlation.
            # If we swap the envelopes, we evaluate corr(pred, wav_b) vs corr(pred, wav_a)
            # The decision rule "predicts A" if corr_first > corr_second.
            # We want to know if it predicted the NEW attended track (which is wav_b).
            if evaluate_trial_majority_vote(pred_base, example.wav_b, example.wav_a, WINDOW_SEC)[0]: cond_correct["Swap Labels"] += 1
            
            # 8. Random Train
            if evaluate_trial_majority_vote(pred_rtrain, example.wav_a, example.wav_b, WINDOW_SEC)[0]: cond_correct["Random Train"] += 1
            
        for k in cond_correct:
            results[held_out][k] = (cond_correct[k] / num_trials) * 100.0
            
    # Print Table
    print("\n=====================================================================================================")
    print(f"{'Subject':<8} | {'Baseline':<8} | {'Zero':<8} | {'Random':<8} | {'Shuffle':<8} | {'Audio':<8} | {'Shift':<8} | {'Swap':<8} | {'RandTrain':<8}")
    print("=====================================================================================================")
    
    metrics = {k: [] for k in results[SUBJECTS_TO_EVAL[0]].keys()}
    
    for sub in SUBJECTS_TO_EVAL:
        r = results[sub]
        for k in metrics:
            metrics[k].append(r[k])
        print(f"{sub:<8} | {r['Baseline']:<8.1f} | {r['Zero EEG']:<8.1f} | {r['Random EEG']:<8.1f} | {r['Shuffle EEG']:<8.1f} | {r['Shuffle Audio']:<8.1f} | {r['Circular Shift']:<8.1f} | {r['Swap Labels']:<8.1f} | {r['Random Train']:<8.1f}")
        
    print("-----------------------------------------------------------------------------------------------------")
    print(f"{'Mean':<8} | {np.mean(metrics['Baseline']):<8.1f} | {np.mean(metrics['Zero EEG']):<8.1f} | {np.mean(metrics['Random EEG']):<8.1f} | {np.mean(metrics['Shuffle EEG']):<8.1f} | {np.mean(metrics['Shuffle Audio']):<8.1f} | {np.mean(metrics['Circular Shift']):<8.1f} | {np.mean(metrics['Swap Labels']):<8.1f} | {np.mean(metrics['Random Train']):<8.1f}")
    print(f"{'Std':<8} | {np.std(metrics['Baseline']):<8.1f} | {np.std(metrics['Zero EEG']):<8.1f} | {np.std(metrics['Random EEG']):<8.1f} | {np.std(metrics['Shuffle EEG']):<8.1f} | {np.std(metrics['Shuffle Audio']):<8.1f} | {np.std(metrics['Circular Shift']):<8.1f} | {np.std(metrics['Swap Labels']):<8.1f} | {np.std(metrics['Random Train']):<8.1f}")
    
    print("\n=====================================================================================================")
    print("INTERPRETATION")
    print("=====================================================================================================")
    mean_base = np.mean(metrics['Baseline'])
    print(f"{'✓' if mean_base > 51 else '✗'} Baseline above chance? ({mean_base:.1f}%) -> {'PASS' if mean_base > 51 else 'FAIL'}")
    
    mean_zero = np.mean(metrics['Zero EEG'])
    print(f"{'✓' if 45 <= mean_zero <= 55 else '✗'} Zero EEG collapsed? ({mean_zero:.1f}%) -> {'PASS' if 45 <= mean_zero <= 55 else 'FAIL'}")
    
    mean_rand = np.mean(metrics['Random EEG'])
    print(f"{'✓' if 45 <= mean_rand <= 55 else '✗'} Random EEG collapsed? ({mean_rand:.1f}%) -> {'PASS' if 45 <= mean_rand <= 55 else 'FAIL'}")
    
    mean_shuff = np.mean(metrics['Shuffle EEG'])
    print(f"{'✓' if 45 <= mean_shuff <= 55 else '✗'} Time shuffle collapsed? ({mean_shuff:.1f}%) -> {'PASS' if 45 <= mean_shuff <= 55 else 'FAIL'}")
    
    mean_aud = np.mean(metrics['Shuffle Audio'])
    print(f"{'✓' if 45 <= mean_aud <= 55 else '✗'} Audio shuffle collapsed? ({mean_aud:.1f}%) -> {'PASS' if 45 <= mean_aud <= 55 else 'FAIL'}")
    
    mean_shift = np.mean(metrics['Circular Shift'])
    print(f"{'✓' if 45 <= mean_shift <= 55 else '✗'} Circular shift collapsed? ({mean_shift:.1f}%) -> {'PASS' if 45 <= mean_shift <= 55 else 'FAIL'}")
    
    mean_swap = np.mean(metrics['Swap Labels'])
    print(f"{'✓' if mean_swap < 50 and abs((100 - mean_base) - mean_swap) < 15 else '✗'} Label swap inverted prediction? ({mean_swap:.1f}%) -> {'PASS' if mean_swap < 50 and abs((100 - mean_base) - mean_swap) < 15 else 'FAIL'}")
    
    mean_rtrain = np.mean(metrics['Random Train'])
    print(f"{'✓' if 45 <= mean_rtrain <= 55 else '✗'} Random-label training collapsed? ({mean_rtrain:.1f}%) -> {'PASS' if 45 <= mean_rtrain <= 55 else 'FAIL'}")

if __name__ == "__main__":
    run_sanity_checks()
