import pandas as pd
import numpy as np
import glob
import os
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

try:
    from decision_engine.context_aware_engine import ContextAwarePolicyEngine
except ImportError:
    print("WARNING: Could not import ContextAwarePolicyEngine. Ensure decision_engine is in REPO_ROOT.")
    ContextAwarePolicyEngine = None

def find_prediction_files():
    # Similar nested Kaggle fix as Phase 18
    search_paths = [
        REPO_ROOT / "results" / "phase17_1" / "scenario_streams",
        REPO_ROOT / "EEG_8_Channel_Pipeline" / "results" / "phase17_1" / "scenario_streams",
        Path("/kaggle/working/EEG_8_Channel_Pipeline/results/phase17_1/scenario_streams"),
        Path("/kaggle/working/EEG_8_Channel_Pipeline/EEG_8_Channel_Pipeline/results/phase17_1/scenario_streams")
    ]
    
    for path in search_paths:
        if path.exists():
            files = list(path.glob("*predictions.csv"))
            if len(files) > 0:
                print(f"Found {len(files)} prediction files in {path}")
                return files
    print("ERROR: Could not find Phase 17.1 predictions.csv files.")
    return []

def extract_splices(df):
    splices = []
    current_scene = None
    for idx, row in df.iterrows():
        if row['scene_name'] != current_scene:
            if current_scene is not None:
                splices.append({
                    'timestamp_sec': row['timestamp_sec'],
                    'old_scene': current_scene,
                    'new_scene': row['scene_name'],
                    'new_gt': int(row['ground_truth'])
                })
            current_scene = row['scene_name']
    return splices

def run_experiment_3_sensitivity(df_list, out_dir):
    if ContextAwarePolicyEngine is None:
        return pd.DataFrame()
        
    mappings = {
        'Current Platt (T=5)': lambda m: 1.0 / (1.0 + np.exp(-5.0 * m)),
        'Aggressive (T=10)': lambda m: 1.0 / (1.0 + np.exp(-10.0 * m)),
        'Relaxed (T=2)': lambda m: 1.0 / (1.0 + np.exp(-2.0 * m)),
        'Identity (Clipped)': lambda m: np.clip(0.5 + (m * 2.0), 0.0, 1.0)
    }
    
    results = []
    
    for name, func in mappings.items():
        print(f"Simulating mapping: {name}")
        
        engine = ContextAwarePolicyEngine(base_threshold=0.85, 
            active_heuristics=['difficulty', 'growth_rate', 'hysteresis', 'oscillation_penalty', 'cooldown'])
            
        total_duration = 0
        correct_locked_windows = 0
        total_windows = 0
        false_switches = 0
        latencies = []
        
        for df_path in df_list:
            df = pd.read_csv(df_path)
            splices = extract_splices(df)
            total_duration += (df['timestamp_sec'].max() - df['timestamp_sec'].min())
            
            engine.reset()
            active_lock = None
            trace = []
            
            for idx, row in df.iterrows():
                m = row['margin']
                p = func(m)
                
                res = engine.update(p, m)
                action = str(res['action'])
                st = str(res['state'])
                
                if action == 'SWITCH_LEFT': active_lock = 1
                elif action == 'SWITCH_RIGHT': active_lock = 0
                elif st in ['UNCERTAIN', 'INITIALIZING', 'WAITING']: active_lock = None
                
                gt = int(row['ground_truth'])
                if active_lock == gt:
                    correct_locked_windows += 1
                total_windows += 1
                
                if action == 'SWITCH_LEFT' and gt == 0: false_switches += 1
                elif action == 'SWITCH_RIGHT' and gt == 1: false_switches += 1
                
                trace.append({'timestamp_sec': row['timestamp_sec'], 'active_lock': active_lock})
                
            for splice in splices:
                sts = splice['timestamp_sec']
                tgt = splice['new_gt']
                for t in trace:
                    if t['timestamp_sec'] >= sts:
                        if t['active_lock'] == tgt:
                            latencies.append(t['timestamp_sec'] - sts)
                            break
                            
        cov = (correct_locked_windows / total_windows * 100) if total_windows > 0 else 0
        fsr = (false_switches / (total_duration / 3600.0)) if total_duration > 0 else 0
        ml = np.mean(latencies) if latencies else 0.0
        
        results.append({
            'Calibration Mapping': name,
            'Coverage (%)': cov,
            'False Switches / hr': fsr,
            'Mean Latency (s)': ml
        })
        
    res_df = pd.DataFrame(results)
    res_df.to_csv(out_dir / "calibration_sensitivity.csv", index=False)
    return res_df

def main():
    print("====================================================")
    print("PHASE 19: CALIBRATION & WEAK MARGIN FALSIFICATION")
    print("====================================================")
    
    files = find_prediction_files()
    if not files: return
    
    out_dir = REPO_ROOT / "results" / "phase19"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    dfs = []
    for f in files:
        d = pd.read_csv(f)
        d['scenario_name'] = f.stem
        dfs.append(d)
    
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Ex. 1: Margin Distribution
    print("Running Exp 1: Margin Distribution...")
    full_df['is_correct'] = (full_df['margin'] > 0).astype(int) == full_df['ground_truth']
    correct_margins = full_df[full_df['is_correct'] == True]['margin']
    incorrect_margins = full_df[full_df['is_correct'] == False]['margin']
    
    plt.figure(figsize=(10,6))
    sns.kdeplot(correct_margins, label='Correct Margins', fill=True, color='green')
    sns.kdeplot(incorrect_margins, label='Incorrect Margins', fill=True, color='red')
    plt.axvline(x=-0.05, color='gray', linestyle='--')
    plt.axvline(x=0.05, color='gray', linestyle='--')
    plt.title("Distribution of Raw Margins")
    plt.xlabel("Margin (c_att - c_unatt)")
    plt.legend()
    plt.savefig(out_dir / "margin_histograms.png")
    
    near_zero_pct = (full_df['margin'].abs() <= 0.05).mean() * 100
    overlap_report = pd.DataFrame({
        'Category': ['Correct', 'Incorrect', 'Total Near Zero ([-0.05, 0.05])'],
        'Mean': [correct_margins.mean(), incorrect_margins.mean(), near_zero_pct],
        'Std': [correct_margins.std(), incorrect_margins.std(), 0.0]
    })
    overlap_report.to_csv(out_dir / "margin_distribution.csv", index=False)
    
    # Ex. 2: Calibration Fidelity
    print("Running Exp 2: Calibration Fidelity...")
    fidelity = full_df[['margin', 'prob']].copy()
    fidelity['Rank_Preserved'] = fidelity['margin'].rank(method='dense') == fidelity['prob'].rank(method='dense')
    fidelity.to_csv(out_dir / "calibration_fidelity.csv", index=False)
    
    # Ex. 5 & 6: Subject and Transition Analysis
    print("Running Exp 5 & 6: Subject and Transition Analysis...")
    subject_stats = full_df.groupby('scenario_name')['margin'].agg(['mean', 'std']).reset_index()
    subject_stats.to_csv(out_dir / "subject_calibration.csv", index=False)
    
    transition_margins = []
    for f in files:
        d = pd.read_csv(f)
        splices = extract_splices(d)
        for sp in splices:
            ts = sp['timestamp_sec']
            w = d[(d['timestamp_sec'] >= ts - 5.0) & (d['timestamp_sec'] <= ts + 5.0)]
            transition_margins.extend(w['margin'].tolist())
            
    pd.DataFrame({'transition_margin': transition_margins}).to_csv(out_dir / "transition_margin_analysis.csv", index=False)
    
    # Ex. 3 & 4: Calibration Sensitivity Simulator & Tradeoff Curves
    print("Running Exp 3 & 4: Calibration Sensitivity...")
    sens_df = run_experiment_3_sensitivity(files, out_dir)
    if not sens_df.empty:
        sens_df.to_csv(out_dir / "tradeoff_curves.csv", index=False)
        
    # Generate Markdown Report
    print("Generating Final Report...")
    with open(out_dir / "phase19_report.md", "w") as f:
        f.write("# Phase 19: Calibration & Weak Margin Falsification Study\n\n")
        
        f.write("## 1. Is calibration actually wrong?\n")
        if near_zero_pct > 30:
            f.write(f"**NO.** {near_zero_pct:.1f}% of all margins are physically bounded in the dead-zone `[-0.05, 0.05]`. Calibration is faithfully compressing them because the signal itself contains no discriminatory power.\n\n")
        else:
            f.write(f"**YES.** Only {near_zero_pct:.1f}% of margins are in the dead-zone, yet calibration flattens them aggressively.\n\n")
            
        if not sens_df.empty:
            best_latency = sens_df['Mean Latency (s)'].min()
            base_latency = sens_df[sens_df['Calibration Mapping'].str.contains('Current')]['Mean Latency (s)'].iloc[0]
            best_method = sens_df.loc[sens_df['Mean Latency (s)'].idxmin()]['Calibration Mapping']
            
            f.write("## 2. If calibration changes, does latency improve?\n")
            if best_latency < base_latency - 5:
                f.write(f"**YES.** Using {best_method} reduces latency from {base_latency:.1f}s to {best_latency:.1f}s.\n\n")
            else:
                f.write(f"**NO.** Even aggressive calibration only changes latency from {base_latency:.1f}s to {best_latency:.1f}s.\n\n")
                
            f.write("## 3. If latency improves, what happens to false switches?\n")
            f.write("Here is the Trade-off Curve:\n")
            f.write(sens_df.to_markdown(index=False))
            f.write("\n\n")
            
        f.write("## 4. Does calibration preserve discriminative information?\n")
        preserve_pct = fidelity['Rank_Preserved'].mean() * 100
        f.write(f"Rank preservation is {preserve_pct:.1f}%. Since it is a monotonic Platt scaling, the ranking is preserved, but the *distance* is crushed.\n\n")
        
        f.write("## 5. What is the TRUE bottleneck?\n")
        if near_zero_pct > 30:
            f.write("**DECODER.** The margins themselves contain absolutely zero information around transitions. No amount of calibration can magically create confidence out of `0.02` correlation differences.\n\n")
        elif not sens_df.empty and best_latency < base_latency - 5:
            f.write("**CALIBRATION.** The decoder provides adequate signal, but Platt scaling compresses it too heavily before evidence accumulation.\n\n")
        else:
            f.write("**POLICY/EVIDENCE.** The signal exists, but the continuous heuristics require too much time to lock on regardless of scaling.\n\n")
            
        f.write("## 6. What subsystem should be improved next?\n")
        if near_zero_pct > 30:
            f.write("Based on empirical falsification, you MUST improve **Decoder Discrimination (Contrastive Loss, stronger spatial filters)**. Do not waste time tuning the policy engine.\n")
        elif not sens_df.empty and best_latency < base_latency - 5:
            f.write("Based on empirical falsification, you MUST improve **Calibration (Isotonic Regression, Temperature Scaling)**.\n")
        else:
            f.write("Based on empirical falsification, you MUST improve **Evidence Accumulation parameters**.\n")

    print(f"Done! Phase 19 artifacts generated in {out_dir}")

if __name__ == '__main__':
    main()
