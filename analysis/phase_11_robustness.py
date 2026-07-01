import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.aad_conformer import AADConformer
from data.kul_cached_dataset import KULCachedLoader
from src.confidence.selective_predictor import SelectivePredictor
from src.confidence.selective_metrics import calculate_selective_risk

def add_noise(eeg_data, snr_db=0):
    """
    Adds Gaussian noise to the EEG signal at a specific SNR.
    """
    eeg_data = np.array(eeg_data)
    signal_power = np.mean(eeg_data ** 2)
    
    # SNR = 10 * log10(signal_power / noise_power)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), eeg_data.shape)
    
    return eeg_data + noise

def test_robustness(subject="S1", model_path=None, threshold=0.70):
    print("=" * 80)
    print("PHASE 11 — SELECTIVE AAD ROBUSTNESS VALIDATION")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        data_path = "/kaggle/input/datasets/lokeshgile/kul-processed/data/processed_kul"
        if not os.path.exists(data_path):
            data_path = "data/processed_kul" # Fallback local
            
        loader = KULCachedLoader(data_path)
        all_data = loader.load_all()
        if subject not in all_data:
            print(f"Subject {subject} not found in data.")
            return
            
        subject_trials = all_data[subject]
        
        # Structure the trials like before
        eeg_clean, wav_a, wav_b, labels = [], [], [], []
        for trial in subject_trials:
            eeg_clean.append(trial["eeg"])
            wav_a.append(trial["audio_a"])
            wav_b.append(trial["audio_b"])
            # KUL cache always sets audio_a to the attended audio, so label is 0
            labels.append(0)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Failed to load data: {e}")
        return
        
    model = AADConformer(in_channels=8).to(device)
    model.eval()
    
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        
    predictor = SelectivePredictor(threshold=threshold)
    
    conditions = [
        ("Clean Baseline", eeg_clean),
        ("Zero EEG", [np.zeros_like(x) for x in eeg_clean]),
        ("Gaussian Noise (SNR=0dB)", [add_noise(x, 0) for x in eeg_clean]),
        ("Gaussian Noise (SNR=-10dB)", [add_noise(x, -10) for x in eeg_clean])
    ]
    
    win_samples = 320
    
    print(f"{'Condition':<25} | {'Coverage':<10} | {'Acc. Acc.':<10} | {'Rej. Acc.':<10} | {'Risk':<10}")
    print("-" * 75)
    
    with torch.no_grad():
        for cond_name, eeg_cond in conditions:
            t_truths = []
            t_preds = []
            t_confs = []
            
            for t_idx in range(len(eeg_cond)):
                t_eeg = eeg_cond[t_idx]
                t_wa = wav_a[t_idx]
                t_wb = wav_b[t_idx]
                label = labels[t_idx]
                
                window_results = []
                
                for start in range(0, t_eeg.shape[1] - win_samples + 1, win_samples):
                    end = start + win_samples
                    c_eeg = torch.FloatTensor(t_eeg[:, start:end]).unsqueeze(0).to(device)
                    c_wa = torch.FloatTensor(t_wa[:, start:end]).unsqueeze(0).to(device)
                    c_wb = torch.FloatTensor(t_wb[:, start:end]).unsqueeze(0).to(device)
                    
                    pred, z_pool = model(c_eeg, return_features=True)
                    
                    pred_c = pred - pred.mean(dim=1, keepdim=True)
                    ya_c = c_wa.squeeze(1) - c_wa.squeeze(1).mean(dim=1, keepdim=True)
                    yb_c = c_wb.squeeze(1) - c_wb.squeeze(1).mean(dim=1, keepdim=True)
                    
                    cov_a = (pred_c * ya_c).sum(dim=1)
                    cov_b = (pred_c * yb_c).sum(dim=1)
                    var_pred = (pred_c ** 2).sum(dim=1)
                    var_a = (ya_c ** 2).sum(dim=1)
                    var_b = (yb_c ** 2).sum(dim=1)
                    
                    sim_a = (cov_a / torch.sqrt(var_pred * var_a + 1e-8)).item()
                    sim_b = (cov_b / torch.sqrt(var_pred * var_b + 1e-8)).item()
                    
                    margin = sim_b - sim_a
                    
                    window_results.append(predictor.predict_window(margin, pearson_a=sim_a, pearson_b=sim_b, use_pearson=True))
                    
                trial_res = predictor.predict_trial(window_results, aggregation="majority")
                
                t_truths.append(label)
                t_preds.append(trial_res["prediction"])
                t_confs.append(trial_res["confidence"])
                
            metrics = calculate_selective_risk(t_truths, t_preds, t_confs, threshold)
            
            c = metrics['coverage'] * 100
            a = metrics['accepted_accuracy'] * 100
            ra = metrics['rejected_accuracy'] * 100
            r = metrics['selective_risk']
            
            print(f"{cond_name:<25} | {c:<9.1f}% | {a:<9.1f}% | {ra:<9.1f}% | {r:<10.4f}")

if __name__ == "__main__":
    model_path = "/kaggle/working/EEG_8_Channel_Pipeline/results/run7_multitask_conformer_loso/checkpoints/seed_1/model_S1.pt"
    test_robustness(subject="S1", model_path=model_path, threshold=0.60)
