import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

try:
    from decision_engine.context_aware_engine import ContextAwarePolicyEngine
    from decision_engine.strategies import (
        InfiniteAccumulator,
        HardCapAccumulator,
        ExponentialDecayAccumulator,
        AsymmetricDecayAccumulator,
        SlidingWindowAccumulator,
        BayesianAccumulator,
        CUSUMHybrid,
        ShiryaevRobertsHybrid,
        PageHinkleyHybrid
    )
except ImportError as e:
    print(f"WARNING: Could not import dependencies: {e}")

def find_prediction_files():
    search_paths = [
        REPO_ROOT / "results" / "phase17_1" / "scenario_streams",
        Path("/kaggle/working/EEG_8_Channel_Pipeline/results/phase17_1/scenario_streams")
    ]
    for path in search_paths:
        if path.exists():
            files = list(path.glob("*predictions.csv"))
            if len(files) > 0:
                return files
    return []

def extract_splices(df):
    splices = []
    current_scene = None
    for idx, row in df.iterrows():
        if row['scene_name'] != current_scene:
            if current_scene is not None:
                splices.append({
                    'timestamp_sec': row['timestamp_sec'],
                    'old_gt': current_scene,
                    'new_gt': int(row['ground_truth'])
                })
            current_scene = row['scene_name']
    return splices

def get_strategies():
    return [
        InfiniteAccumulator(),
        HardCapAccumulator(cap=150.0),
        ExponentialDecayAccumulator(decay=0.99),
        AsymmetricDecayAccumulator(decay=0.85),
        SlidingWindowAccumulator(window_size=128),
        BayesianAccumulator(p_switch=0.0001),
        CUSUMHybrid(drift=0.5, threshold=3.0),
        ShiryaevRobertsHybrid(threshold=20.0),
        PageHinkleyHybrid(delta=0.1, threshold=5.0)
    ]

def evaluate_strategy(strategy, files):
    total_metrics = []
    
    for f in files:
        df = pd.read_csv(f)
        scenario = f.stem
        splices = extract_splices(df)
        
        # Instantiate engine with this specific strategy
        engine = ContextAwarePolicyEngine(
            base_threshold=0.85, 
            active_heuristics=['difficulty', 'growth_rate', 'hysteresis', 'oscillation_penalty', 'cooldown'],
            strategy=strategy
        )
        
        trace = []
        for idx, row in df.iterrows():
            res = engine.update(row['prob'], row['margin'])
            trace.append({
                'timestamp_sec': row['timestamp_sec'],
                'window_idx': idx, 
                'ground_truth': int(row['ground_truth']),
                'cumulative_evidence': res['evidence'], 
                'confidence': res['confidence'],
                'active_threshold': res['threshold_used'], 
                'state': res['state'],
                'decision': res['decision'], 
                'active_lock': 1 if res['action'] == 'SWITCH_LEFT' else (0 if res['action'] == 'SWITCH_RIGHT' else None)
            })
            
        trace_df = pd.DataFrame(trace)
        trace_df['active_lock'] = trace_df['active_lock'].ffill()
        
        # Evaluate Metrics (Acquisition, Switch, Coverage, etc)
        # Using Phase 17 standard
        stats = engine.statistics()
        
        # Compute latencies
        acquisition_latencies = []
        switch_latencies = []
        
        # Initial acquisition
        if not splices:
            # Single scenario
            lock_row = trace_df[trace_df['active_lock'] == df.iloc[0]['ground_truth']]
            if not lock_row.empty:
                acquisition_latencies.append(lock_row.iloc[0]['timestamp_sec'] - df.iloc[0]['timestamp_sec'])
        else:
            # Splice handling
            lock_row = trace_df[trace_df['active_lock'] == df.iloc[0]['ground_truth']]
            if not lock_row.empty and lock_row.iloc[0]['timestamp_sec'] < splices[0]['timestamp_sec']:
                acquisition_latencies.append(lock_row.iloc[0]['timestamp_sec'] - df.iloc[0]['timestamp_sec'])
                
            for sp in splices:
                ts = sp['timestamp_sec']
                tgt = sp['new_gt']
                post_splice = trace_df[trace_df['timestamp_sec'] >= ts]
                lock = post_splice[post_splice['active_lock'] == tgt]
                if not lock.empty:
                    switch_latencies.append(lock.iloc[0]['timestamp_sec'] - ts)
                    
        # Coverage
        correct = trace_df[trace_df['active_lock'] == trace_df['ground_truth']]
        wrong = trace_df[(trace_df['active_lock'].notna()) & (trace_df['active_lock'] != trace_df['ground_truth'])]
        avail = trace_df[trace_df['active_lock'].notna()]
        
        total_time = trace_df['timestamp_sec'].max() - trace_df['timestamp_sec'].min()
        
        total_metrics.append({
            'Strategy': strategy.get_name(),
            'Scenario': scenario,
            'Acquisition_Latency_s': np.mean(acquisition_latencies) if acquisition_latencies else np.nan,
            'Switch_Latency_s': np.mean(switch_latencies) if switch_latencies else np.nan,
            'Correct_Coverage_Pct': len(correct) / len(trace_df) * 100,
            'Wrong_Coverage_Pct': len(wrong) / len(trace_df) * 100,
            'Availability_Pct': len(avail) / len(trace_df) * 100,
            'Oscillations': stats['oscillations'],
            'Mean_Lock_Duration_s': stats['avg_lock_duration'] * 0.0625,
            'Peak_Evidence': trace_df['cumulative_evidence'].abs().max()
        })
        
    return pd.DataFrame(total_metrics)

def main():
    print("====================================================")
    print("PHASE 22: CONTINUOUS DECISION STRATEGY BENCHMARK")
    print("====================================================")
    
    files = find_prediction_files()
    if not files: return
    
    out_dir = REPO_ROOT / "results" / "phase22"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    strategies = get_strategies()
    print(f"Benchmarking {len(strategies)} temporal decision strategies...")
    
    all_results = []
    
    for strategy in tqdm(strategies, desc="Strategies"):
        df = evaluate_strategy(strategy, files)
        all_results.append(df)
        
    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(out_dir / "strategy_metrics.csv", index=False)
    
    # Aggregate across scenarios
    agg_df = final_df.groupby('Strategy').agg({
        'Acquisition_Latency_s': 'mean',
        'Switch_Latency_s': 'mean',
        'Correct_Coverage_Pct': 'mean',
        'Wrong_Coverage_Pct': 'mean',
        'Availability_Pct': 'mean',
        'Oscillations': 'mean',
        'Mean_Lock_Duration_s': 'mean',
        'Peak_Evidence': 'max'
    }).reset_index()
    
    agg_df.to_csv(out_dir / "strategy_rankings.csv", index=False)
    print("Benchmarking complete. Artifacts generated.")
    
if __name__ == '__main__':
    main()
