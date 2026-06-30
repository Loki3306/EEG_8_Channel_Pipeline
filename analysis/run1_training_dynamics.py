import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "conformer_loso"
SUMMARY_FILE = RESULTS_DIR / "conformer_loso_multiseed_summary.json"

def pad_sequences(seq_list, max_len=15):
    padded = np.full((len(seq_list), max_len), np.nan)
    for i, seq in enumerate(seq_list):
        length = min(len(seq), max_len)
        padded[i, :length] = seq[:length]
    return padded

def plot_dynamics(data_matrix, ylabel, title, out_path, color='blue'):
    means = np.nanmean(data_matrix, axis=0)
    stds = np.nanstd(data_matrix, axis=0)
    epochs = np.arange(1, len(means) + 1)
    
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, means, color=color, linewidth=2, label='Mean')
    plt.fill_between(epochs, means - stds, means + stds, color=color, alpha=0.2, label='±1 Std Dev')
    
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(epochs)
    
    plt.savefig(out_path, dpi=300)
    plt.close()

def main():
    if not SUMMARY_FILE.exists():
        print(f"Error: {SUMMARY_FILE} not found.")
        return
        
    with open(SUMMARY_FILE, "r") as f:
        data = json.load(f)
        
    all_train_losses = []
    all_val_losses = []
    all_val_margins = []
    stopped_epochs = []
    best_epochs = []
    
    for seed, seed_data in data.items():
        for subject, metrics in seed_data.items():
            if "history" in metrics:
                hist = metrics["history"]
                all_train_losses.append(hist.get("train_loss", []))
                all_val_losses.append(hist.get("val_loss", []))
                all_val_margins.append(hist.get("val_margin", []))
            
            stopped_epochs.append(metrics.get("stopped_epoch", 15))
            best_epochs.append(metrics.get("best_epoch", -1))
            
    if not all_train_losses:
        print("No history data found in the summary JSON.")
        return
        
    train_loss_mat = pad_sequences(all_train_losses)
    val_loss_mat = pad_sequences(all_val_losses)
    val_margin_mat = pad_sequences(all_val_margins)
    
    out_dir = RESULTS_DIR / "figures"
    out_dir.mkdir(exist_ok=True, parents=True)
    
    # Plot Dynamics
    plot_dynamics(train_loss_mat, "Train Loss", "Training Loss Dynamics", out_dir / "dyn_train_loss.png", color='blue')
    plot_dynamics(val_loss_mat, "Validation Loss", "Validation Loss Dynamics", out_dir / "dyn_val_loss.png", color='orange')
    plot_dynamics(val_margin_mat, "Validation Margin", "Validation Margin Dynamics", out_dir / "dyn_val_margin.png", color='green')
    
    # Early Stopping Histogram
    plt.figure(figsize=(8, 6))
    bins = np.arange(0.5, 16.5, 1)
    plt.hist(stopped_epochs, bins=bins, alpha=0.6, color='purple', label='Stopped Epoch')
    plt.hist(best_epochs, bins=bins, alpha=0.6, color='gold', label='Best Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Frequency')
    plt.title('Distribution of Best and Early Stopping Epochs')
    plt.xticks(np.arange(1, 16))
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(out_dir / "dyn_stopping_hist.png", dpi=300)
    plt.close()
    
    print(f"Saved training dynamics plots to {out_dir}")

if __name__ == "__main__":
    main()
