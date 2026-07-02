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

def load_eeg_sample(path, dataset_type='aasd'):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find {path}")
        
    if path.endswith('.mat'):
        mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
        eeg_var = [k for k in mat.keys() if not k.startswith('__')][0]
        data = mat[eeg_var]
        if hasattr(data, 'data'):
            data = data.data
        elif isinstance(data, np.ndarray):
            pass # Keep as is
            
        if dataset_type == 'aasd':
            # AASD: (62, 7680, 60) -> Downsample to 64Hz
            data = data[:, ::2, :]
            win = data[:8, :128, 0]
        else:
            # Assume KUL format
            if data.ndim == 3:
                win = data[:8, :128, 0]
            elif data.ndim == 2:
                win = data[:8, :128]
            else:
                win = data.flatten()[:8*128].reshape(8, 128)
                
        return torch.tensor(win, dtype=torch.float32).unsqueeze(0)
        
    elif path.endswith('.npy'):
        data = np.load(path)
        if data.ndim == 3:
            win = data[0, :8, :128]
        else:
            win = data[:8, :128]
        return torch.tensor(win, dtype=torch.float32).unsqueeze(0)
    else:
        raise ValueError("Unsupported file format")

def run_audit(kul_path, aasd_path, checkpoint_path):
    print("====================================================")
    print("PHASE 24.6: ACTIVATION DISTRIBUTION AUDIT")
    print("====================================================")
    
    model = AADConformer(in_channels=8)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            # Standard PyTorch load
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            model.load_state_dict(state_dict)
            print(f"[INFO] Loaded checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"[WARN] Failed to load checkpoint: {e}. Using random weights.")
    else:
        print("[WARN] No valid checkpoint provided. Using randomly initialized weights.")
        
    model.eval()
    
    # Register hooks
    if hasattr(model, 'spatial_conv'):
        model.spatial_conv.register_forward_hook(get_hook('1_Stem'))
    if hasattr(model, 'conformer_blocks') and len(model.conformer_blocks) > 0:
        model.conformer_blocks[0].register_forward_hook(get_hook('2_Block1'))
        model.conformer_blocks[-1].register_forward_hook(get_hook('3_BlockN'))
        
    datasets = {}
    if kul_path:
        try:
            datasets['KUL'] = load_eeg_sample(kul_path, 'kul')
            print(f"[PASS] Loaded KUL sample from {kul_path}")
        except Exception as e:
            print(f"[FAIL] KUL load error: {e}")
            
    if aasd_path:
        try:
            datasets['AASD'] = load_eeg_sample(aasd_path, 'aasd')
            print(f"[PASS] Loaded AASD sample from {aasd_path}")
        except Exception as e:
            print(f"[FAIL] AASD load error: {e}")

    if not datasets:
        print("[FAIL] No datasets loaded. Exiting.")
        return

    results = {}
    for name, x in datasets.items():
        activations.clear()
        x_np = x.numpy()
        activations['0_Input'] = x_np
        
        with torch.no_grad():
            try:
                out = model(x)
                activations['4_Output'] = out.detach().cpu().numpy()
            except Exception as e:
                print(f"[FAIL] Forward pass crashed for {name}: {e}")
                continue
                
        results[name] = {}
        for layer, act in sorted(activations.items()):
            mean = np.mean(act)
            std = np.std(act)
            zeros = np.sum(act == 0.0) / act.size * 100
            results[name][layer] = {'mean': mean, 'std': std, 'zeros': zeros}
            
    print("\n--- DISTRIBUTION COMPARISON ---")
    layers = sorted(list(results.get('KUL', results.get('AASD', {})).keys()))
    
    header = f"{'Layer':<15} | {'KUL Mean':<10} | {'KUL Std':<10} | {'KUL %0':<8} || {'AASD Mean':<10} | {'AASD Std':<10} | {'AASD %0':<8}"
    print(header)
    print("-" * len(header))
    
    for layer in layers:
        k = results.get('KUL', {}).get(layer, {'mean': np.nan, 'std': np.nan, 'zeros': np.nan})
        a = results.get('AASD', {}).get(layer, {'mean': np.nan, 'std': np.nan, 'zeros': np.nan})
        
        print(f"{layer:<15} | {k['mean']:>10.4f} | {k['std']:>10.4f} | {k['zeros']:>7.1f}% || {a['mean']:>10.4f} | {a['std']:>10.4f} | {a['zeros']:>7.1f}%")
        
    print("\n[NOTE] If '0_Input' std differs massively, it's a preprocessing mismatch.")
    print("[NOTE] If '3_BlockN' std collapses (<0.01) on AASD but not KUL, it's a domain representation shift.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kul_path", type=str, default="", help="Path to KUL sample")
    parser.add_argument("--aasd_path", type=str, default="", help="Path to AASD sample")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to Conformer checkpoint")
    args = parser.parse_args()
    run_audit(args.kul_path, args.aasd_path, args.checkpoint)
