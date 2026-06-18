"""
analyze_s1_raw_structure.py
Stage-1 Forensic Analysis for S1.mat
Goal: Discover structure, keys, shapes, and potential datasets without loading
everything into RAM. No assumptions, no defaults, no PSD.
"""

import os
import sys
import argparse
import scipy.io as sio
import h5py
import numpy as np
from pathlib import Path

def print_and_write(f_out, text):
    print(text)
    f_out.write(text + "\n")

def analyze_scipy_mat(mat, f_out):
    print_and_write(f_out, "=== MATLAB Format: <= v7.2 (Loaded with scipy.io) ===")
    
    candidates = {'trials': [], 'labels': [], 'fsample': []}
    
    def traverse(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k.startswith('__'): continue
                shape = getattr(v, 'shape', 'scalar')
                dtype = getattr(v, 'dtype', type(v).__name__)
                
                print_and_write(f_out, f"{prefix}- Key: {k} | Type: {type(v).__name__} | Shape: {shape} | Dtype: {dtype}")
                check_candidates(k, v, shape)
                traverse(v, prefix + "  ")
                
        elif hasattr(obj, '_fieldnames'):  # mat_struct
            for k in obj._fieldnames:
                v = getattr(obj, k)
                shape = getattr(v, 'shape', 'scalar')
                dtype = getattr(v, 'dtype', type(v).__name__)
                
                print_and_write(f_out, f"{prefix}- Field: {k} | Type: {type(v).__name__} | Shape: {shape} | Dtype: {dtype}")
                check_candidates(k, v, shape)
                
                # Do not recurse into massive arrays
                if not isinstance(v, np.ndarray) or (isinstance(v, np.ndarray) and v.size < 100):
                    traverse(v, prefix + "  ")

    def check_candidates(k, v, shape):
        name = str(k).lower()
        if 'trial' in name and 'info' not in name:
            candidates['trials'].append((k, shape))
        elif 'label' in name:
            candidates['labels'].append((k, shape))
        elif 'fsample' in name or 'srate' in name:
            val = "Unknown"
            if isinstance(v, (int, float)):
                val = v
            elif isinstance(v, np.ndarray) and v.size == 1:
                val = v.flat[0]
            candidates['fsample'].append((k, val))

    traverse(mat)
    
    print_and_write(f_out, "\n=== Candidate Discoveries ===")
    print_and_write(f_out, f"Potential Trial arrays: {candidates['trials']}")
    print_and_write(f_out, f"Potential Label arrays: {candidates['labels']}")
    print_and_write(f_out, f"Potential Sampling Rates: {candidates['fsample']}")


def analyze_h5py_mat(f_mat, f_out):
    print_and_write(f_out, "=== MATLAB Format: v7.3 (Loaded with h5py) ===")
    
    candidates = {'trials': [], 'labels': [], 'fsample': []}
    
    def traverse(name, node):
        shape = getattr(node, 'shape', 'scalar')
        dtype = getattr(node, 'dtype', type(node).__name__)
        size_bytes = getattr(node, 'nbytes', 0)
        
        print_and_write(f_out, f"- {name} | Type: {type(node).__name__} | Shape: {shape} | Dtype: {dtype} | Size: {size_bytes} bytes")
        
        base_name = name.split('/')[-1].lower()
        
        # Trial candidates
        if 'trial' in base_name and 'info' not in base_name:
            if isinstance(node, h5py.Dataset):
                candidates['trials'].append((name, shape, dtype))
                
        # Label candidates
        elif 'label' in base_name:
            if isinstance(node, h5py.Dataset):
                candidates['labels'].append((name, shape, dtype))
                # Attempt to extract first label as a diagnostic
                try:
                    if node.dtype.kind == 'O':  # Object references
                        ref = node[0, 0] if len(node.shape) > 1 else node[0]
                        obj = f_mat[ref]
                        lbl = ''.join(chr(c[0]) for c in obj[:])
                        candidates['labels'].append(f"  [Diagnostic: First label decoded as '{lbl}']")
                except Exception as e:
                    candidates['labels'].append(f"  [Diagnostic: Label decoding failed: {e}]")
                    
        # Fsample candidates
        elif 'fsample' in base_name or 'srate' in base_name:
            if isinstance(node, h5py.Dataset) and node.size == 1:
                try:
                    val = node[()]
                    if isinstance(val, np.ndarray): val = val.flat[0]
                    candidates['fsample'].append((name, val))
                except Exception:
                    candidates['fsample'].append((name, "Extraction failed"))

    f_mat.visititems(traverse)
    
    print_and_write(f_out, "\n=== Candidate Discoveries ===")
    print_and_write(f_out, "Potential Trial arrays:")
    for t in candidates['trials']: print_and_write(f_out, f"  {t}")
        
    print_and_write(f_out, "\nPotential Label arrays:")
    for l in candidates['labels']: print_and_write(f_out, f"  {l}")
        
    print_and_write(f_out, "\nPotential Sampling Rates:")
    for fs in candidates['fsample']: print_and_write(f_out, f"  {fs}")


def run_analysis(mat_path, out_dir):
    mat_path = Path(mat_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting forensic analysis on {mat_path}")
    
    struct_report_path = out_dir / "s1_structure.txt"
    
    with open(struct_report_path, "w") as f_out:
        print_and_write(f_out, f"File: {mat_path.name}")
        print_and_write(f_out, f"Size: {mat_path.stat().st_size / (1024**2):.2f} MB\n")
        
        try:
            mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
            analyze_scipy_mat(mat, f_out)
        except NotImplementedError:
            print_and_write(f_out, "scipy.io failed. Attempting h5py (MATLAB v7.3)...")
            try:
                with h5py.File(mat_path, 'r') as f_mat:
                    analyze_h5py_mat(f_mat, f_out)
            except Exception as e:
                print_and_write(f_out, f"FATAL ERROR loading with h5py: {e}")
        except Exception as e:
            print_and_write(f_out, f"FATAL ERROR loading with scipy.io: {e}")

    print(f"\nAnalysis complete. Structure report saved to {struct_report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1_path", type=str, default="/kaggle/input/raw-eeh/S1.mat", help="Path to raw S1.mat")
    parser.add_argument("--out_dir", type=str, default="/kaggle/working/reports", help="Directory to save reports")
    args = parser.parse_args()
    
    run_analysis(args.s1_path, args.out_dir)
