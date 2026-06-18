"""
frequency_band_benchmark.py
Efficient Mini-LOSO frequency band benchmark using the validated MatchNet pipeline.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import welch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from training.train_matchnet_loso import train_matchnet_loso
from baselines.ridge_aad import subject_files, load_subject_examples

FS = 64.0
MINI_LOSO_SUBJECTS = ['S1', 'S4', 'S6', 'S8', 'S11', 'S14']
CHANNELS = [13, 46, 43, 23, 50, 0, 52, 14]

VARIANTS = {
    'A': {'low': 0.1, 'high': None, 'name': '0.1 Hz HP (Baseline)'},
    'B': {'low': 1.0, 'high': 8.0, 'name': '1-8 Hz (Current)'},
    'C': {'low': 1.0, 'high': 12.0, 'name': '1-12 Hz'},
    'D': {'low': 4.0, 'high': 8.0, 'name': '4-8 Hz (Theta)'},
    'E': {'low': 8.0, 'high': 12.0, 'name': '8-12 Hz (Alpha)'},
    'F': {'low': 12.0, 'high': 30.0, 'name': '12-30 Hz (Beta)'},
    'G': {'low': 1.0, 'high': 30.0, 'name': '1-30 Hz'},
    'H': {'low': 4.0, 'high': 30.0, 'name': '4-30 Hz'}
}

def print_and_write(f_out, text):
    print(text)
    f_out.write(text + "\n")

def compute_psd_stats(eeg_data):
    all_psds = []
    f_psd = None
    for eeg in eeg_data:
        f, pxx = welch(eeg, fs=FS, nperseg=int(FS*2), axis=0) # eeg is (time, channels)
        if f_psd is None: f_psd = f
        all_psds.append(np.mean(pxx, axis=1)) # mean over channels -> (freq,)
    
    avg_psd = np.mean(all_psds, axis=0)
    
    total_power = np.trapezoid(avg_psd[f_psd <= 30], f_psd[f_psd <= 30])
    bands = {'Delta': (1,4), 'Theta': (4,8), 'Alpha': (8,12), 'Beta': (12,30)}
    stats = {}
    for name, (low, high) in bands.items():
        idx = np.logical_and(f_psd >= low, f_psd <= high)
        power = np.trapezoid(avg_psd[idx], f_psd[idx])
        stats[name] = (power / total_power) * 100 if total_power > 0 else 0
    return stats

def build_benchmark(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "frequency_band_benchmark_report.md"
    csv_path = out_dir / "frequency_band_results.csv"
    
    print("Preloading MAT dataset into RAM to avoid disk I/O...")
    all_paths = subject_files()
    if not all_paths:
        print("ERROR: No subjects found.")
        return
        
    subject_cache = {}
    psd_stats = []
    for p in all_paths:
        sub_key = p.stem.split("_")[0]
        if sub_key in MINI_LOSO_SUBJECTS:
            print(f"  Loading {sub_key}...")
            examples = load_subject_examples(p)
            subject_cache[str(p)] = examples
            
            # PSD on unfiltered EEG (which is time, channels)
            eegs = [ex.eeg[:, CHANNELS] for ex in examples]
            stats = compute_psd_stats(eegs)
            stats['Subject'] = sub_key
            psd_stats.append(stats)
            
    print("\nStarting Mini-LOSO Training using VALIDATED MatchNet Pipeline...")
    results = []
    
    for v_key, v_params in VARIANTS.items():
        v_name = v_params['name']
        print(f"\nEvaluating Variant {v_key}: {v_name}")
        
        norm_dict, zero_dict, shuf_dict = train_matchnet_loso(
            eeg_model="vlaai_lite",
            channels=CHANNELS,
            lowcut=v_params['low'],
            highcut=v_params['high'],
            batch_size=256,
            num_workers=4,
            target_subjects=MINI_LOSO_SUBJECTS,
            subject_examples_cache=subject_cache
        )
        
        # norm_dict is { 2: [acc_s1, acc_s4...], 5: [acc_s1, acc_s4...] }
        for w_sec in [2, 5]:
            if w_sec in norm_dict:
                accs = norm_dict[w_sec]
                for acc in accs:
                    results.append({
                        'Variant': v_name,
                        'Window': w_sec,
                        'Accuracy': acc * 100
                    })
            
    # Compile Results
    df_res = pd.DataFrame(results)
    
    # Calculate means
    mean_res_2s = df_res[df_res['Window'] == 2].groupby('Variant')['Accuracy'].mean().reset_index().rename(columns={'Accuracy': 'Acc_2s'})
    mean_res_5s = df_res[df_res['Window'] == 5].groupby('Variant')['Accuracy'].mean().reset_index().rename(columns={'Accuracy': 'Acc_5s'})
    
    mean_res = pd.merge(mean_res_2s, mean_res_5s, on='Variant', how='outer')
    mean_res['Mean_Acc'] = (mean_res['Acc_2s'] + mean_res['Acc_5s']) / 2.0
    
    # Calculate Delta using 1-8 Hz variant
    if '1-8 Hz (Current)' in mean_res['Variant'].values:
        baseline_mean = mean_res.loc[mean_res['Variant'] == '1-8 Hz (Current)', 'Mean_Acc'].values[0]
        mean_res['Delta'] = mean_res['Mean_Acc'] - baseline_mean
    else:
        mean_res['Delta'] = 0.0
        
    # Output to CSV
    mean_res.to_csv(csv_path, index=False)
    
    # Generate Report
    with open(report_path, "w") as f_out:
        print_and_write(f_out, "# Frequency-Band Benchmark Report\n")
        
        print_and_write(f_out, "## 1. PSD Power Distribution")
        df_psd = pd.DataFrame(psd_stats)
        print_and_write(f_out, df_psd.to_markdown(index=False))
        print_and_write(f_out, "\n")
        
        print_and_write(f_out, "## 2. Benchmark Results (Mini-LOSO MatchNet Pipeline)")
        print_and_write(f_out, mean_res[['Variant', 'Acc_2s', 'Acc_5s', 'Mean_Acc', 'Delta']].to_markdown(index=False))
        print_and_write(f_out, "\n")
        
        # Success Criteria
        best_delta = mean_res['Delta'].max()
        best_variant = mean_res.loc[mean_res['Delta'].idxmax(), 'Variant']
        
        print_and_write(f_out, "## 3. Recommendation")
        if best_delta >= 2.0:
            print_and_write(f_out, f"✅ **SUCCESS**: Variant '{best_variant}' improved performance by +{best_delta:.2f}%. Recommend promoting to full LOSO evaluation.")
        else:
            print_and_write(f_out, f"❌ **FAILURE**: No frequency variant met the +2.0% threshold (Best: {best_variant} at {best_delta:+.2f}%). Recommend stopping further frequency-band exploration.")
            
    print(f"\nBenchmark complete. Report saved to {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="reports")
    args = parser.parse_args()
    build_benchmark(args.out_dir)
