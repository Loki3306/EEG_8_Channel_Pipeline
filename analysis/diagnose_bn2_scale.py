import os
import sys
import pickle
import numpy as np

def analyze():
    dtu_cache_path = 'data/DTU_Micro_Activations.pkl'
    kul_cache_path = 'data/KUL_Micro_Activations.pkl'
    
    if not os.path.exists(dtu_cache_path) or not os.path.exists(kul_cache_path):
        print("Caches missing. Please run E11 to generate.")
        return
        
    with open(dtu_cache_path, 'rb') as f:
        dtu_act = pickle.load(f)
    with open(kul_cache_path, 'rb') as f:
        kul_act = pickle.load(f)
        
    l3 = "03_SpatialConv"
    l4 = "04_BatchNorm2"
    
    print("="*60)
    print("DIAGNOSTIC: 03_SpatialConv vs 04_BatchNorm2")
    print("="*60)
    
    for layer in [l3, l4]:
        d = dtu_act[layer]
        k = kul_act[layer]
        
        mu_d = np.mean(d, axis=0)
        mu_k = np.mean(k, axis=0)
        
        diff = np.abs(mu_d - mu_k)
        
        sig_d = np.var(d, axis=0)
        sig_k = np.var(k, axis=0)
        
        print(f"\n{layer}")
        print(f"Mean (DTU): {mu_d[:4]} ...")
        print(f"Mean (KUL): {mu_k[:4]} ...")
        print(f"Mean Diff: {diff[:4]} ...")
        print(f"Max Mean Diff: {np.max(diff):.6f}")
        print(f"Var (DTU): {sig_d[:4]} ...")
        print(f"Var (KUL): {sig_k[:4]} ...")
        
        # Norm of mean difference
        l2_diff = np.sum((mu_d - mu_k)**2)
        print(f"L2 Squared distance of Means (||mu_D - mu_K||^2): {l2_diff:.6f}")
        
    # Check the actual scaling ratio between L3 and L4
    d3 = dtu_act[l3]
    d4 = dtu_act[l4]
    
    # Estimate the BN scaling factor (A)
    # var(Y) = A^2 var(X)  => A = sqrt(var(Y)/var(X))
    var3 = np.var(d3, axis=0)
    var4 = np.var(d4, axis=0)
    scale_A = np.sqrt(var4 / (var3 + 1e-12))
    
    print("\n" + "="*60)
    print("Estimated BatchNorm2 Scaling Factors (Gamma / sigma)")
    print("="*60)
    print(f"Scale A: {scale_A[:8]} ...")
    print(f"Max Scale: {np.max(scale_A):.2f}")
    print(f"Min Scale: {np.min(scale_A):.2f}")
    
    # If Max Scale is large, it explains why the L2 Squared distance explodes.
    # Because L2_Y = A^2 * L2_X. If A = 10, A^2 = 100.
    
if __name__ == "__main__":
    analyze()
