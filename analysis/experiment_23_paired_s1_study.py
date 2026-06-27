import os
import sys
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.matchnet import ContrastiveMatchNet, contrastive_loss
from train_kul_native import get_kul_trials, preprocess_trial, chunk_data, evaluate_val
from training.train_matchnet_loso import prepare_dataset, chunk_trial, evaluate_model, get_mapping_data
from baselines.ridge_aad import load_subject_examples, subject_files

def cka_linear(act1, act2):
    act1 = act1 - act1.mean(dim=0, keepdim=True)
    act2 = act2 - act2.mean(dim=0, keepdim=True)
    dot_prod = torch.norm(torch.mm(act1.t(), act2), p='fro') ** 2
    norm_1 = torch.norm(torch.mm(act1.t(), act1), p='fro')
    norm_2 = torch.norm(torch.mm(act2.t(), act2), p='fro')
    return (dot_prod / (norm_1 * norm_2)).item()

def extract_features(model, bx, bya, byb):
    features = {}
    
    # 1. Spatial Conv out
    x = model.eeg_encoder.block1(bx)
    features['spatial_conv'] = x.view(x.size(0), -1)
    
    # 2. Block 2 out
    x = model.eeg_encoder.block2(x)
    features['temporal_conv'] = x.view(x.size(0), -1)
    
    # 3. Final embedding
    z_eeg = model.encode_eeg(bx)
    features['embedding'] = z_eeg.view(z_eeg.size(0), -1)
    
    return features

def train_dtu_s1(device):
    print("\n--- Training DTU S1 Model ---")
    all_paths = subject_files()
    s1_path = [p for p in all_paths if "S1_data" in p.name]
    if not s1_path:
        print("DTU S1 data not found!")
        return None
    s1_path = s1_path[0]
    
    examples = load_subject_examples(s1_path)
    mapping, envelopes = get_mapping_data()
    
    np.random.seed(42)
    np.random.shuffle(examples)
    val_split = int(0.2 * len(examples))
    val_exs = examples[:val_split]
    train_exs = examples[val_split:]
    
    channels = [0, 1, 3, 5, 8, 12, 17, 19] # MatchNet 8 channels (T7, C2, etc.)
    lowcut, highcut = 1.0, 8.0
    
    print(f"Loaded {len(train_exs)} Train Trials, {len(val_exs)} Val Trials.")
    
    X_tr_full, YA_tr_full, YB_tr_full = prepare_dataset(train_exs, channels, lowcut, highcut, s1_path.stem, mapping, envelopes)
    X_va_full, YA_va_full, YB_va_full = prepare_dataset(val_exs, channels, lowcut, highcut, s1_path.stem, mapping, envelopes)
    
    X_tr, YA_tr, YB_tr = [], [], []
    for i in range(len(X_tr_full)):
        cx, cya, cyb = chunk_trial(X_tr_full[i], YA_tr_full[i], YB_tr_full[i], window_sec=5, hop_sec=2)
        X_tr.extend(cx); YA_tr.extend(cya); YB_tr.extend(cyb)
        
    print(f"Generated {len(X_tr)} DTU training chunks.")
    
    X_tr_t = torch.FloatTensor(np.stack(X_tr))
    YA_tr_t = torch.FloatTensor(np.stack(YA_tr))
    YB_tr_t = torch.FloatTensor(np.stack(YB_tr))
    
    train_loader = DataLoader(TensorDataset(X_tr_t, YA_tr_t, YB_tr_t), batch_size=128, shuffle=True)
    
    model = ContrastiveMatchNet("eegnet", eeg_channels=8, audio_channels=28, latent_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    
    best_val_acc = 0.0
    best_weights = deepcopy(model.state_dict())
    patience = 5
    epochs_no_improve = 0
    
    for epoch in range(50):
        model.train()
        for bx, bya, byb in train_loader:
            bx, bya, byb = bx.to(device), bya.to(device), byb.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                z_eeg, z_a, z_b = model(bx, bya, byb)
                loss, _, _ = contrastive_loss(z_eeg, z_a, z_b)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        nc, nt = evaluate_model(model, X_va_full, YA_va_full, YB_va_full, device, window_sec=10)
        val_acc = nc / max(nt, 1)
        
        print(f"Epoch {epoch+1:02d} | Val Acc (10s): {val_acc*100:.2f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
                
    model.load_state_dict(best_weights)
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/matchnet_dtu_s1_best.pth")
    return model, X_va_full, YA_va_full, YB_va_full

def train_kul_s1(device):
    print("\n--- Training KUL S1 Model ---")
    
    cache_path = "kul_gammatone_cache.pkl"
    if os.path.exists(cache_path):
        import pickle
        with open(cache_path, "rb") as f:
            envelope_cache = pickle.load(f)
    else:
        print("Missing kul_gammatone_cache.pkl")
        return None, None
        
    trials = get_kul_trials()
    train_trials = trials[:15]
    val_trials = trials[15:]
    
    tr_x_chunks, tr_ya_chunks, tr_yb_chunks = [], [], []
    val_data = []
    
    for t in train_trials:
        x, ya, yb = preprocess_trial(t, envelope_cache, apply_car=True)
        if x is not None:
            cx, cya, cyb = chunk_data(x, ya, yb, window_sec=5, hop_sec=2)
            tr_x_chunks.extend(cx)
            tr_ya_chunks.extend(cya)
            tr_yb_chunks.extend(cyb)
            
    for t in val_trials:
        x, ya, yb = preprocess_trial(t, envelope_cache, apply_car=True)
        if x is not None:
            val_data.append((x, ya, yb))
            
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(np.stack(tr_x_chunks)), 
                                            torch.FloatTensor(np.stack(tr_ya_chunks)), 
                                            torch.FloatTensor(np.stack(tr_yb_chunks))), 
                              batch_size=128, shuffle=True)
                              
    model = ContrastiveMatchNet("eegnet", eeg_channels=8, audio_channels=28, latent_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    
    best_val_acc = 0.0
    best_weights = deepcopy(model.state_dict())
    patience = 5
    epochs_no_improve = 0
    
    for epoch in range(50):
        model.train()
        for bx, bya, byb in train_loader:
            bx, bya, byb = bx.to(device), bya.to(device), byb.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                z_eeg, z_a, z_b = model(bx, bya, byb)
                loss, _, _ = contrastive_loss(z_eeg, z_a, z_b)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        val_acc = evaluate_val(model, val_data, device, window_sec=10)
        
        print(f"Epoch {epoch+1:02d} | Val Acc (10s): {val_acc*100:.2f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
                
    model.load_state_dict(best_weights)
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/matchnet_kul_s1_best.pth")
    return model, val_data

def run_experiment_23():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Train Both Models
    dtu_model, dtu_val_x, dtu_val_ya, dtu_val_yb = train_dtu_s1(device)
    kul_model, kul_val_data = train_kul_s1(device)
    
    if not dtu_model or not kul_model: return
    
    # 2. Cross-Evaluate (using DTU evaluate_model and KUL evaluate_val)
    print("\n================================================================================")
    print("PAIRED CROSS-DATASET MATRIX (S1 ONLY)")
    print("================================================================================")
    
    nc, nt = evaluate_model(dtu_model, dtu_val_x, dtu_val_ya, dtu_val_yb, device, window_sec=10)
    dd_acc = nc / max(nt, 1)
    
    dk_acc = evaluate_val(dtu_model, kul_val_data, device, window_sec=10)
    
    kk_acc = evaluate_val(kul_model, kul_val_data, device, window_sec=10)
    
    nc, nt = evaluate_model(kul_model, dtu_val_x, dtu_val_ya, dtu_val_yb, device, window_sec=10)
    kd_acc = nc / max(nt, 1)
    
    print(f"| {'Train':<10} | {'Test':<10} | {'Accuracy':<10} |")
    print("-" * 38)
    print(f"| {'DTU S1':<10} | {'DTU S1':<10} | {dd_acc*100:>9.2f}% |")
    print(f"| {'DTU S1':<10} | {'KUL S1':<10} | {dk_acc*100:>9.2f}% |")
    print(f"| {'KUL S1':<10} | {'KUL S1':<10} | {kk_acc*100:>9.2f}% |")
    print(f"| {'KUL S1':<10} | {'DTU S1':<10} | {kd_acc*100:>9.2f}% |")
    print("================================================================================")
    
    # 3. CKA Computation (Using KUL Validation Data)
    print("\nComputing Representation CKA (using KUL S1 test data)...")
    dtu_model.eval(); kul_model.eval()
    
    all_dtu_features = {'spatial_conv': [], 'temporal_conv': [], 'embedding': []}
    all_kul_features = {'spatial_conv': [], 'temporal_conv': [], 'embedding': []}
    
    with torch.no_grad():
        for x, ya, yb in kul_val_data:
            start = 0
            while start + 320 <= x.shape[1]:
                end = start + 320
                cx = torch.FloatTensor(x[:, start:end]).unsqueeze(0).to(device)
                cya = torch.FloatTensor(ya[:, start:end]).unsqueeze(0).to(device)
                cyb = torch.FloatTensor(yb[:, start:end]).unsqueeze(0).to(device)
                
                f_dtu = extract_features(dtu_model, cx, cya, cyb)
                f_kul = extract_features(kul_model, cx, cya, cyb)
                
                for k in f_dtu.keys():
                    all_dtu_features[k].append(f_dtu[k])
                    all_kul_features[k].append(f_kul[k])
                    
                start += 320
                
    for k in all_dtu_features.keys():
        f1 = torch.cat(all_dtu_features[k], dim=0)
        f2 = torch.cat(all_kul_features[k], dim=0)
        cka_score = cka_linear(f1, f2)
        print(f"  CKA Layer [{k:<15}]: {cka_score:.4f}")
        
    # 4. Plot Filters
    dtu_spatial = dtu_model.eeg_encoder.block1[2].weight.detach().cpu().numpy().squeeze()
    kul_spatial = kul_model.eeg_encoder.block1[2].weight.detach().cpu().numpy().squeeze()
    channels = ['T7', 'C2', 'FT8', 'P7', 'CPz', 'Fp1', 'TP8', 'C3']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    vmin = -max(np.abs(dtu_spatial).max(), np.abs(kul_spatial).max())
    vmax = -vmin
    
    im1 = axes[0].imshow(dtu_spatial, aspect='auto', cmap='coolwarm', vmin=vmin, vmax=vmax)
    axes[0].set_title("DTU S1 Optimal Spatial Filters")
    axes[0].set_xticks(np.arange(8)); axes[0].set_xticklabels(channels)
    axes[0].set_yticks(np.arange(16)); axes[0].set_ylabel("Filter Index")
    
    im2 = axes[1].imshow(kul_spatial, aspect='auto', cmap='coolwarm', vmin=vmin, vmax=vmax)
    axes[1].set_title("KUL S1 Optimal Spatial Filters")
    axes[1].set_xticks(np.arange(8)); axes[1].set_xticklabels(channels)
    axes[1].set_yticks(np.arange(16))
    
    fig.colorbar(im1, ax=axes.ravel().tolist(), label="Weight")
    plt.savefig("s1_paired_filters.png")
    plt.close()
    print("\nSaved 's1_paired_filters.png'.")

if __name__ == "__main__":
    run_experiment_23()
