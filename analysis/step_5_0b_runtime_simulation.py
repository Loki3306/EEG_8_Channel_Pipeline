import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure src can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.confidence.inference_engine import ConfidenceEngine

def main():
    print("===========================================")
    print("STEP 5.0b: END-TO-END RUNTIME SIMULATION")
    print("===========================================")
    
    model_path = "models/confidence_model.json"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Please run step_5_0a_train_final_model.py first.")
        return
        
    engine = ConfidenceEngine(model_path=model_path, threshold=0.80)
    
    # Load data for simulation
    csv_path = "subject_distance_predictions.csv"
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Cannot find {csv_path}.")
        return
        
    df = df[np.isfinite(df['margin'])].reset_index(drop=True)
    df = df.sort_values(['subject_id', 'trial_id', 'window_id']).reset_index(drop=True)
    
    # Extract the very first trial
    first_subject = df['subject_id'].iloc[0]
    first_trial = df[(df['subject_id'] == first_subject)]['trial_id'].iloc[0]
    
    trial_data = df[(df['subject_id'] == first_subject) & (df['trial_id'] == first_trial)].copy()
    print(f"Simulating streaming inference for Subject {first_subject}, Trial {first_trial} ({len(trial_data)} windows)...\n")
    
    engine.reset_trial()
    
    results = []
    
    for i, row in trial_data.iterrows():
        # Simulate receiving similarities
        sim_a = row['sim_A']
        sim_b = row['sim_B']
        
        res = engine.predict_with_confidence(eeg_window=None, sim_a=sim_a, sim_b=sim_b)
        
        results.append({
            'window_id': row['window_id'],
            'margin': res['margin'],
            'confidence': res['confidence'],
            'prediction': res['prediction'],
            'accept': res['accept'],
            'correct': row['correct']
        })
        
        acc_str = "ACCEPT" if res['accept'] else "REJECT"
        print(f"Window {row['window_id']:<4} | Pred: {res['prediction']} | Margin: {res['margin']:>6.3f} | Conf: {res['confidence']:>5.3f} | {acc_str}")
        
    res_df = pd.DataFrame(results)
    
    # Generate Plots
    os.makedirs('analysis/figures', exist_ok=True)
    
    time_axis = res_df['window_id']
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # 1. Margin vs Time
    axs[0].plot(time_axis, res_df['margin'], 'b-', marker='o', markersize=3, label='Margin')
    axs[0].axhline(0, color='k', linestyle='--', alpha=0.5)
    axs[0].set_ylabel('MatchNet Margin')
    axs[0].set_title('Margin over Time')
    axs[0].grid(True, alpha=0.3)
    
    # 2. Confidence vs Time
    axs[1].plot(time_axis, res_df['confidence'], 'g-', marker='s', markersize=3, label='Confidence')
    axs[1].axhline(0.80, color='r', linestyle='--', label='Accept Threshold (0.8)')
    axs[1].set_ylabel('P(Correct)')
    axs[1].set_title('XGBoost Confidence over Time')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    
    # 3. Accept / Reject Timeline
    # We plot the actual correctness, colored by Accept/Reject
    accepted = res_df[res_df['accept']]
    rejected = res_df[~res_df['accept']]
    
    axs[2].scatter(accepted['window_id'], accepted['correct'], color='green', marker='^', s=100, label='Accepted')
    axs[2].scatter(rejected['window_id'], rejected['correct'], color='red', marker='x', s=100, label='Rejected')
    axs[2].set_yticks([0, 1])
    axs[2].set_yticklabels(['Incorrect', 'Correct'])
    axs[2].set_xlabel('Window ID')
    axs[2].set_title('Decision Timeline vs Ground Truth Correctness')
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis/figures/runtime_simulation_5_0.png', dpi=300)
    plt.close()
    
    print(f"\nPlots generated at analysis/figures/runtime_simulation_5_0.png")
    print("\nSimulation complete. Operational pipeline is ready.")

if __name__ == "__main__":
    main()
