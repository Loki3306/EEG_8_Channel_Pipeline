import torch
import numpy as np
import scipy.io
import argparse
import sys
import os
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from models.aad_conformer import AADConformer
except ImportError:
    print("Could not import AADConformer.")
    sys.exit(1)

activations = {}

def get_hook(name):
    def hook(model, input, output):
        activations[name] = output.detach().cpu().numpy()
    return hook

def load_kul(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find {path}")
    data = torch.load(path, map_location='cpu')
    win = None
    if isinstance(data, dict):
        if 'trials' in data and isinstance(data['trials'], list) and len(data['trials']) > 0:
            trial = data['trials'][0]
            if 'eeg' in trial: win = trial['eeg']
            elif 'data' in trial: win = trial['data']
        else:
            if 'eeg' in data: win = data['eeg']
            elif 'data' in data: win = data['data']
            else:
                for k, v in data.items():
                    if isinstance(v, torch.Tensor) and v.ndim >= 2:
                        win = v
                        break
    elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        trial = data[0]
        if 'eeg' in trial: win = trial['eeg']
        elif 'data' in trial: win = trial['data']
    else:
        win = data
        
    if isinstance(win, torch.Tensor):
        if win.ndim == 3: win = win[0, :8, :128]
        elif win.ndim == 2: win = win[:8, :128]
        else: win = win.view(-1)[:8*128].reshape(8, 128)
        return win.clone().detach().to(torch.float32).unsqueeze(0)
    raise ValueError("Could not parse KUL data")

def load_aasd(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find {path}")
    mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    eeg_var = [k for k in mat.keys() if not k.startswith('__')][0]
    data = mat[eeg_var]
    if hasattr(data, 'data'):
        data = data.data
        
    # data: (62, 7680, 60)
    data_64 = data[:, ::2, :]
    
    global_mean = float(np.mean(data_64))
    global_std = float(np.std(data_64))
    
    win = data_64[:8, :128, 0]
    win = torch.tensor(win, dtype=torch.float32)
    
    return win, global_mean, global_std

def run_audit(kul_path, aasd_path, checkpoint_path):
    print("====================================================")
    print("PHASE 24.7: NORMALIZATION AUDIT")
    print("====================================================")
    
    model = AADConformer(in_channels=8)
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            model.load_state_dict(state_dict, strict=False)
            print(f"[INFO] Loaded checkpoint (strict=False)")
        except Exception as e:
            print(f"[WARN] Failed to load checkpoint: {e}. Using random weights.")
    else:
        print("[WARN] No checkpoint provided. Using random weights.")
            
    model.eval()
    
    if hasattr(model, 'spatial_conv'):
        model.spatial_conv.register_forward_hook(get_hook('1_Stem'))
    if hasattr(model, 'conformer_blocks') and len(model.conformer_blocks) > 0:
        model.conformer_blocks[-1].register_forward_hook(get_hook('2_BlockN'))
        
    print("[INFO] Loading datasets...")
    kul_win = load_kul(kul_path)
    aasd_win, global_mean, global_std = load_aasd(aasd_path)
    print("[PASS] Datasets loaded successfully.")
    
    # 4 Variants
    aasd_raw = aasd_win.clone().unsqueeze(0)
    
    aasd_global = (aasd_win.clone() - global_mean) / global_std
    aasd_global = aasd_global.unsqueeze(0)
    
    trial_mean = aasd_win.mean()
    trial_std = aasd_win.std()
    aasd_trial = (aasd_win.clone() - trial_mean) / trial_std
    aasd_trial = aasd_trial.unsqueeze(0)
    
    chan_mean = aasd_win.mean(dim=1, keepdim=True)
    chan_std = aasd_win.std(dim=1, keepdim=True)
    aasd_channel = (aasd_win.clone() - chan_mean) / (chan_std + 1e-6)
    aasd_channel = aasd_channel.unsqueeze(0)
    
    datasets = {
        'KUL (Target)': kul_win,
        'AASD_Raw': aasd_raw,
        'AASD_Global': aasd_global,
        'AASD_Trial': aasd_trial,
        'AASD_Channel': aasd_channel
    }
    
    results = {}
    for name, x in datasets.items():
        activations.clear()
        activations['0_Input'] = x.numpy()
        
        with torch.no_grad():
            try:
                out = model(x)
                activations['3_Output'] = out.detach().cpu().numpy()
            except Exception as e:
                print(f"[FAIL] {name} forward crashed: {e}")
                continue
                
        results[name] = {}
        for layer, act in sorted(activations.items()):
            mean = np.mean(act)
            std = np.std(act)
            results[name][layer] = {'mean': mean, 'std': std}
            
    print("\n--- DISTRIBUTION COMPARISON (KUL VS AASD VARIANTS) ---")
    layers = sorted(list(results.get('KUL (Target)', {}).keys()))
    
    for layer in layers:
        print(f"\n[{layer}]")
        target_std = results['KUL (Target)'][layer]['std']
        
        for name in datasets.keys():
            res = results.get(name, {}).get(layer, {'mean': np.nan, 'std': np.nan})
            diff = abs(res['std'] - target_std)
            if name == 'KUL (Target)':
                print(f"  {name:<15} | Mean: {res['mean']:>8.4f} | Std: {res['std']:>8.4f} | Target (Base)")
            else:
                print(f"  {name:<15} | Mean: {res['mean']:>8.4f} | Std: {res['std']:>8.4f} | Abs Diff to KUL: {diff:>8.4f}")
                
    print("\n[CONCLUSION]")
    print("Look at the 'Abs Diff to KUL' for '2_BlockN'. The variant with the smallest difference is your optimal adapter.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kul_path", type=str, required=True, help="Path to KUL sample")
    parser.add_argument("--aasd_path", type=str, required=True, help="Path to AASD sample")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to Conformer checkpoint")
    args = parser.parse_args()
    run_audit(args.kul_path, args.aasd_path, args.checkpoint)
