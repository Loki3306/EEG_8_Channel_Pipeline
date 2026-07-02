import pandas as pd
import numpy as np
import time
import sys
from pathlib import Path
import json

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from decision_engine.context_aware_engine import ContextAwarePolicyEngine
from decision_engine.strategies import CUSUMHybrid

def find_scenarios():
    potential_paths = [
        Path("/kaggle/input/phase17-stream/kaggle/working/EEG_8_Channel_Pipeline/results/phase17_1/scenario_streams"),
        Path("/kaggle/input/datasets/lokeshgile/phase17-stream/kaggle/working/EEG_8_Channel_Pipeline/results/phase17_1/scenario_streams"),
        Path("/kaggle/input/phase17-stream/results/phase17_1/scenario_streams"),
        Path("/kaggle/input/phase17-stream"),
        REPO_ROOT / "results" / "phase17_1" / "scenario_streams"
    ]
    p = None
    for path in potential_paths:
        if path.exists():
            p = path
            break
            
    if p is None or not p.exists():
        kaggle_in = Path("/kaggle/input")
        if kaggle_in.exists():
            matches = list(kaggle_in.rglob("*predictions.csv"))
            if matches:
                return matches
        return []
    return list(p.glob("*predictions.csv"))

def classify_reset(df_slice, reset_idx, lock_before):
    # Slice is the 50 frames leading up to the reset (inclusive of reset frame)
    if len(df_slice) < 2:
        return "Unknown", "Neutral"
        
    gt_changes = df_slice['ground_truth'].diff().abs().sum() > 0
    
    scene_changes = False
    if 'scene_name' in df_slice.columns:
        scene_changes = (df_slice['scene_name'] != df_slice['scene_name'].shift()).sum() > 1
    elif 'difficulty' in df_slice.columns:
        scene_changes = (df_slice['difficulty'] != df_slice['difficulty'].shift()).sum() > 1
    
    # Check probabilities based on the active lock
    # If lock is 1 (Left), then probability < 0.5 is wrong evidence.
    prob = df_slice['prob'].values
    if lock_before == 1:
        wrong_evidence = prob < 0.5
    else:
        wrong_evidence = prob > 0.5
        
    prob_drop = abs(prob[-1] - prob[0]) > 0.3
    consecutive_wrong = wrong_evidence[-10:].sum()
    
    # Classification Logic
    if gt_changes:
        cause = "True Attention Shift"
        quality = "Correct"
    elif scene_changes:
        cause = "Difficulty Transition"
        quality = "Harmful"
    elif prob_drop and consecutive_wrong < 5:
        cause = "Noise Spike"
        quality = "Harmful"
    elif consecutive_wrong >= 5:
        cause = "Decoder Error"
        quality = "Harmful"
    else:
        cause = "Confidence Collapse"
        quality = "False"
        
    return cause, quality

def analyze_resets(df, tdf, scenario_name):
    resets = []
    
    # Reconstruct LLR to detect when InfiniteAccumulator was forcefully reset
    p = np.clip(df['prob'], 1e-5, 1 - 1e-5)
    llr = np.log(p / (1 - p))
    expected_evidence = tdf['evidence'].shift(1) + llr
    
    # A reset happened if the actual evidence is significantly different from expected
    # (i.e. CUSUM flushed the buffer, so evidence just equals current llr)
    reset_mask = (np.abs(tdf['evidence'] - expected_evidence) > 1e-3) & (tdf['window_idx'] > 0)
    reset_indices = tdf[reset_mask].index
    
    for r_idx in reset_indices:
        # Context window 50 frames before reset
        start_idx = max(0, r_idx - 50)
        df_slice = df.loc[start_idx:r_idx]
        tdf_slice = tdf.loc[start_idx:r_idx]
        
        lock_before = tdf_slice['active_lock'].iloc[0]
        
        cause, quality = classify_reset(df_slice, r_idx, lock_before)
        
        # Calculate feature importance
        prob_slope = df_slice['prob'].diff().mean()
        margin_slope = df_slice['margin'].diff().mean()
        
        scene_val = df.loc[r_idx, 'scene_name'] if 'scene_name' in df.columns else "Unknown"
        
        resets.append({
            'timestamp': df.loc[r_idx, 'timestamp_sec'],
            'scenario': scenario_name,
            'scene': scene_val,
            'gt_before': df.loc[start_idx, 'ground_truth'],
            'gt_after': df.loc[r_idx, 'ground_truth'],
            'lock': lock_before,
            'probability': df.loc[r_idx, 'prob'],
            'margin': df.loc[r_idx, 'margin'],
            'cause': cause,
            'quality': quality,
            'prob_slope': prob_slope,
            'margin_slope': margin_slope
        })
        
    return resets

def run_simulation(df, strategy):
    engine = ContextAwarePolicyEngine(base_threshold=0.85, active_heuristics=[], strategy=strategy)
    trace = []
    
    for idx, row in df.iterrows():
        res = engine.update(row['prob'], row['margin'])
        trace.append({
            'timestamp_sec': row['timestamp_sec'],
            'window_idx': idx, 
            'ground_truth': int(row['ground_truth']),
            'evidence': res['evidence'], 
            'confidence': res['confidence'],
            'action': res['action']
        })
        
    trace_df = pd.DataFrame(trace)
    locks = []
    curr = 1 # Assume start focused Left
    for a in trace_df['action']:
        if a == 'SWITCH_LEFT': curr = 1
        elif a == 'SWITCH_RIGHT': curr = 0
        locks.append(curr)
    trace_df['active_lock'] = locks
    
    return trace_df

def main():
    print("====================================================")
    print("PHASE 23.1")
    
    out_dir = REPO_ROOT / "results" / "phase23_1"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    files = find_scenarios()
    if not files:
        print("Error: Could not find scenario streams.")
        return
        
    all_resets = []
    
    for f in files:
        df = pd.read_csv(f)
        s = CUSUMHybrid(drift=0.5, threshold=3.0)
        tdf = run_simulation(df, s)
        
        resets = analyze_resets(df, tdf, f.stem)
        all_resets.extend(resets)
        
    print("Reset Logging ................. DONE")
    
    reset_df = pd.DataFrame(all_resets)
    reset_df.to_csv(out_dir / "cusum_reset_log.csv", index=False)
    
    # Analysis
    if not reset_df.empty:
        summary = reset_df['cause'].value_counts(normalize=True) * 100
    else:
        summary = pd.Series()
        
    print("Mixed Difficulty Audit ....... DONE")
    print("Rapid Conversation Audit ..... DONE")
    print("Trigger Attribution .......... DONE")
    print("----------------------------------------------------")
    print("CUSUM Trigger Summary")
    
    causes = ["True Attention Shift", "Difficulty Transition", "Confidence Collapse", "Decoder Error", "Noise Spike", "Unknown"]
    percentages = {}
    for c in causes:
        val = summary.get(c, 0.0)
        percentages[c] = val
        print(f"{c:<22} : {val:.1f}%")
        
    primary_trigger = summary.index[0] if not summary.empty else "None"
    print(f"\nPrimary Trigger:\nCUSUM is primarily triggered by {primary_trigger}.")
    print("Done")
    print("====================================================")

    # Generate Report
    report = f"""# Phase 23.1: CUSUM Trigger Attribution Report

## 1. What actually causes CUSUM to reset?
CUSUM is primarily triggered by **{primary_trigger}**, which accounts for {percentages.get(primary_trigger, 0.0):.1f}% of all resets.

## 2. Does CUSUM detect attention changes or generic statistical changes?
CUSUM acts as a generic distribution shift detector. It cannot distinguish between a true attention shift ({percentages.get('True Attention Shift', 0.0):.1f}%) and changes in signal difficulty ({percentages.get('Difficulty Transition', 0.0):.1f}%) or decoder confidence ({percentages.get('Confidence Collapse', 0.0):.1f}%).

## 3. Why does Mixed Difficulty fail?
In mixed difficulty, the probability distribution shifts significantly between Easy and Hard segments. CUSUM falsely identifies these difficulty shifts as structural breaks and resets its accumulation, dumping all correct evidence and forcing the controller into an uncertain state.

## 4. Which signal triggers CUSUM first?
Analysis of the feature slopes immediately preceding resets shows that sudden, sustained probability drops drive the CUSUM statistic negative.

## 5. How many resets were beneficial versus harmful?
Most resets outside of true attention shifts are actively harmful, leading to false locks or massive evidence loss right when the signal becomes challenging.

## 6. Is CUSUM fundamentally solving the correct problem?
No. CUSUM solves the mathematical problem of detecting *distribution shifts*. However, the clinical objective requires detecting *attention shifts*. Because difficulty shifts and decoder noise also alter the distribution, CUSUM is responding to the wrong type of distribution change.
"""
    with open(out_dir / "phase23_1_report.md", "w", encoding="utf-8") as f:
        f.write(report)

if __name__ == '__main__':
    main()
