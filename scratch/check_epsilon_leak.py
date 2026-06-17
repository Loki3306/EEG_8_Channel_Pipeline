import numpy as np

def normalize_array(arr):
    arr = arr - arr.mean(axis=0, keepdims=True)
    scale = arr.std(axis=0, keepdims=True) + 1e-12
    return arr / scale

np.random.seed(42)
# Stream A is quiet (std = 0.1)
a = np.random.randn(100, 28) * 0.1
# Stream B is loud (std = 1.0)
b = np.random.randn(100, 28) * 1.0

a_norm = normalize_array(a)
b_norm = normalize_array(b)

var_a = np.var(a_norm, axis=0)
var_b = np.var(b_norm, axis=0)

diff = var_a - var_b
print(f"Variance A: {var_a[0]:.15f}")
print(f"Variance B: {var_b[0]:.15f}")
print(f"Difference: {diff[0]:.15e}")
