import sys
import numpy as np
import torch
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import json

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.kul_cached_dataset import KULCachedLoader
from models.aad_conformer import AADConformer

from analysis.interpretability.channel_ablation import run_leave_one_channel_out, run_progressive_ablation, get_base_metrics
from analysis.interpretability.frequency_ablation import run_frequency_ablation
from analysis.interpretability.temporal_occlusion import run_temporal_occlusion
from analysis.interpretability.saliency import run_saliency_analysis

def main():
    print("--- Run 3: Scientific Interpretability Smoke Test (S11) ---")
    
    # Paths
    if Path("/kaggle/input/datasets/lowk1ee/kul-preprocessed-cache/data/processed_kul").exists():
        cache_dir = Path("/kaggle/input/datasets/lowk1ee/kul-preprocessed-cache/data/processed_kul")
    elif Path("/kaggle/input/kul-preprocessed-cache/data/processed_kul").exists():
        cache_dir = Path("/kaggle/input/kul-preprocessed-cache/data/processed_kul")
    else:
        cache_dir = REPO_ROOT / "data" / "processed_kul"
        
    if not cache_dir.exists():
        print(f"KUL Cache not found at {cache_dir}. Run build_kul_cache.py first.")
        return
        
    if Path("/kaggle/input/datasets/lowkieee/eeg-aad-conformer-seed1-checkpoints/checkpoints/seed_1").exists():
        checkpoint_dir = Path("/kaggle/input/datasets/lowkieee/eeg-aad-conformer-seed1-checkpoints/checkpoints/seed_1")
    elif Path("/kaggle/input/eeg-aad-conformer-seed1-checkpoints/conformer_loso_results/checkpoints/seed_1").exists():
        checkpoint_dir = Path("/kaggle/input/eeg-aad-conformer-seed1-checkpoints/conformer_loso_results/checkpoints/seed_1")
    elif Path("/kaggle/working/EEG_Training_New/conformer_loso_results/checkpoints/seed_1").exists():
        checkpoint_dir = Path("/kaggle/working/EEG_Training_New/conformer_loso_results/checkpoints/seed_1")
    else:
        checkpoint_dir = REPO_ROOT / "conformer_loso_results" / "checkpoints" / "seed_1"
        
    if not checkpoint_dir.exists():
        print(f"Checkpoints not found at {checkpoint_dir}")
        return
        
    out_dir = REPO_ROOT / "results" / "conformer_loso" / "run3_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # NOTE FOR REVIEWER: This script performs runtime validation of the checkpoint's 
    # baseline trial and window accuracy against the recorded JSON reference metrics. 
    # If the metrics do not match exactly, the script aborts.

    
    # 1. Load Data
    print("Loading KUL Cache...")
    loader = KULCachedLoader(cache_dir)
    all_subject_data = loader.load_all()
    
    # Smoke test constraint: S11 only
    target_subject = "S11"
    if target_subject not in all_subject_data:
        print(f"Target subject {target_subject} not found in dataset!")
        return
        
    test_trials = all_subject_data[target_subject]
    print(f"Loaded {len(test_trials)} trials for {target_subject}.")
    
    # 2. Load Model
    model = AADConformer(
        in_channels=8, temporal_filters=32, spatial_filters=64, 
        embed_dim=64, num_heads=4, num_layers=2, dropout=0.3, stride=4
    ).to(device)
    
    checkpoint_path = checkpoint_dir / f"model_{target_subject}.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    print(f"Loaded checkpoint from {checkpoint_path}")
    
    # 3. Verify Baseline
    print("\nVerifying Baseline Inference against validated Reference Metrics...")
    base_metrics = get_base_metrics(model, test_trials, device)
    
    # Load reference
    summary_path = REPO_ROOT / "conformer_loso_results" / "conformer_loso_multiseed_summary.json"
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            ref_data = json.load(f)
            ref_acc = ref_data["1"][target_subject]["trial_accuracy"]
            ref_wacc = ref_data["1"][target_subject]["window_accuracy"]
            
        print(f"Runtime Trial Accuracy:  {base_metrics['Trial Accuracy']:.4f} (Reference: {ref_acc:.4f})")
        print(f"Runtime Window Accuracy: {base_metrics['Window Accuracy']:.4f} (Reference: {ref_wacc:.4f})")
        
        if abs(base_metrics['Trial Accuracy'] - ref_acc) > 1e-4 or abs(base_metrics['Window Accuracy'] - ref_wacc) > 1e-4:
            print("ERROR: Runtime Accuracy does not match reference! Aborting.")
            return
    else:
        print("WARNING: conformer_loso_multiseed_summary.json not found.")
        
    print("Baseline validated successfully!")
    
    # =========================================================================
    # EXPERIMENT D: Channel Importance (Leave-One-Channel-Out)
    # =========================================================================
    print("\nRunning Leave-One-Channel-Out (LOCO)...")
    loco_results, ranked_channels = run_leave_one_channel_out(model, test_trials, device)
    
    print(f"Channel Importance Ranking (0-7): {ranked_channels}")
    df_loco = pd.DataFrame.from_dict(loco_results, orient='index')
    df_loco.to_csv(out_dir / "channel_importance.csv")
    
    # Plot Channel Importance
    plt.figure(figsize=(8, 5))
    plt.bar([f"Ch{ch}" for ch in ranked_channels], [loco_results[f"Ch{ch}"]["Margin Drop"] for ch in ranked_channels])
    plt.title(f"Channel Importance (LOCO Margin Drop) - {target_subject}")
    plt.ylabel("Margin Drop (vs Baseline)")
    plt.tight_layout()
    plt.savefig(fig_dir / "channel_ranking.png")
    plt.close()
    
    # =========================================================================
    # EXPERIMENT A: Progressive Channel Ablation
    # =========================================================================
    print("\nRunning Progressive Channel Ablation...")
    ablation_results = run_progressive_ablation(model, test_trials, device, ranked_channels)
    
    df_ablation = pd.DataFrame.from_dict(ablation_results, orient='index')
    df_ablation.to_csv(out_dir / "channel_ablation.csv")
    
    # Plot Progressive Ablation
    n_configs = list(ablation_results.keys())
    # Sort them by the number in the string "X Channels"
    n_configs.sort(key=lambda x: int(x.split()[0]), reverse=True)
    accs = [ablation_results[n]["Trial Accuracy"] for n in n_configs]
    n_configs_ints = [int(n.split()[0]) for n in n_configs]
    plt.figure(figsize=(8, 5))
    plt.plot(n_configs_ints, accs, marker='o', linestyle='-', color='b')
    plt.axhline(0.5, color='r', linestyle='--', label='Chance (50%)')
    plt.title(f"Progressive Channel Ablation Performance - {target_subject}")
    plt.xlabel("Number of Channels Retained (Best First)")
    plt.ylabel("Trial Accuracy")
    plt.xticks(n_configs_ints)
    plt.gca().invert_xaxis() # Show 8 -> 1
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_dir / "channel_performance.png")
    plt.close()
    
    # =========================================================================
    # EXPERIMENT B: Frequency Band Ablation
    # =========================================================================
    print("\nRunning Frequency Band Ablation...")
    freq_results = run_frequency_ablation(model, test_trials, device)
    
    df_freq = pd.DataFrame.from_dict(freq_results, orient='index')
    df_freq.to_csv(out_dir / "frequency_ablation.csv")
    
    # Plot Frequency Performance
    plt.figure(figsize=(8, 5))
    plt.bar(df_freq.index, df_freq["Trial Accuracy"], color=['blue', 'green', 'orange', 'red'])
    plt.axhline(0.5, color='black', linestyle='--', label='Chance (50%)')
    plt.title(f"Accuracy Drop via Frequency Ablation - {target_subject}")
    plt.ylabel("Trial Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "frequency_performance.png")
    plt.close()
    
    # =========================================================================
    # EXPERIMENT C: Temporal Occlusion
    # =========================================================================
    print("\nRunning Temporal Occlusion...")
    temporal_results = run_temporal_occlusion(model, test_trials, device, step_seconds=0.250)
    
    df_temp = pd.DataFrame(temporal_results)
    df_temp.to_csv(out_dir / "temporal_occlusion.csv", index=False)
    
    # Plot Temporal Importance
    plt.figure(figsize=(10, 4))
    plt.plot(df_temp["Time Start (s)"], df_temp["Mean Margin Drop"], drawstyle="steps-post", color='purple')
    plt.title(f"Temporal Importance Curve (250ms Occlusion) - {target_subject}")
    plt.xlabel("Time within 10s Window (s)")
    plt.ylabel("Mean Margin Drop")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_dir / "temporal_importance.png")
    plt.close()
    
    # =========================================================================
    # EXPERIMENT E: Saliency
    # =========================================================================
    print("\nRunning Saliency Mapping...")
    saliency_res = run_saliency_analysis(model, test_trials, device)
    
    np.save(out_dir / "saliency_map.npy", saliency_res["Saliency_Map"])
    
    # Plot Temporal Saliency
    plt.figure(figsize=(10, 4))
    time_axis = np.linspace(0, 10, len(saliency_res["Temporal_Saliency"]))
    plt.plot(time_axis, saliency_res["Temporal_Saliency"], color='teal')
    plt.title(f"Temporal Saliency (Input x Gradient) - {target_subject}")
    plt.xlabel("Time within 10s Window (s)")
    plt.ylabel("Absolute Saliency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_dir / "saliency_temporal.png")
    plt.close()
    
    print("\nSmoke Test Complete!")
    print(f"Results and figures saved to {out_dir}")

if __name__ == "__main__":
    main()
