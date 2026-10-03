import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Union

class StreamingCATCNEngine:
    """
    Optimized deployment inference engine for CA-TCN.
    
    Supports:
      - Native PyTorch FP32
      - TorchScript compilation (torch.jit.trace)
      - Dynamic INT8 Quantization (torch.ao.quantization.quantize_dynamic)
      - Precise inference latency profiling (T_compute)
    """
    def __init__(self, model: nn.Module, mode: str = "torchscript", 
                 device: str = "cpu", dummy_window_samples: int = 320,
                 num_threads: Optional[int] = 4):
        self.device = torch.device(device)
        self.mode = mode.lower()
        if self.device.type == "cpu" and num_threads is not None:
            try:
                torch.set_num_threads(num_threads)
            except Exception:
                pass
        self.model = model.to(self.device).eval()
        
        # Verify model architecture parameters
        eeg_channels = getattr(model, "eeg_encoder", None)
        in_ch = eeg_channels.spatial_proj.in_channels if hasattr(eeg_channels, "spatial_proj") else 8
        self.in_channels = in_ch
        
        # Prepare execution backend
        dummy_eeg = torch.zeros(1, in_ch, dummy_window_samples, device=self.device)
        dummy_a = torch.zeros(1, 1, dummy_window_samples, device=self.device)
        dummy_b = torch.zeros(1, 1, dummy_window_samples, device=self.device)
        
        if self.mode == "torchscript":
            with torch.no_grad():
                self.runner = torch.jit.trace(self.model, (dummy_eeg, dummy_a, dummy_b))
        elif self.mode == "pytorch_int8":
            # Dynamic INT8 quantization for CPU linear/conv layers
            quantized = torch.ao.quantization.quantize_dynamic(
                self.model.cpu(), {nn.Linear}, dtype=torch.qint8
            )
            self.runner = quantized.eval()
            self.device = torch.device("cpu")
        elif self.mode == "pytorch_fp32":
            self.runner = self.model
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose 'torchscript', 'pytorch_int8', or 'pytorch_fp32'.")
            
        # Warmup pass
        with torch.no_grad():
            self.predict(dummy_eeg.cpu().numpy(), dummy_a.cpu().numpy(), dummy_b.cpu().numpy())

    @torch.no_grad()
    def predict(self, eeg: Union[np.ndarray, torch.Tensor],
                audio_a: Union[np.ndarray, torch.Tensor],
                audio_b: Union[np.ndarray, torch.Tensor]) -> dict:
        """
        Executes single-window forward inference.
        
        Parameters:
            eeg: [1, n_channels, T] or [n_channels, T]
            audio_a: [1, 1, T] or [T]
            audio_b: [1, 1, T] or [T]
            
        Returns:
            dict with 'delta', 'logit_a', 'logit_b', 'compute_ms'
        """
        if isinstance(eeg, np.ndarray):
            if eeg.ndim == 2:
                eeg = eeg[np.newaxis, :, :]
            eeg_t = torch.from_numpy(eeg).float().to(self.device)
        else:
            eeg_t = eeg.to(self.device)
            if eeg_t.ndim == 2:
                eeg_t = eeg_t.unsqueeze(0)
                
        if isinstance(audio_a, np.ndarray):
            if audio_a.ndim == 1:
                audio_a = audio_a[np.newaxis, np.newaxis, :]
            elif audio_a.ndim == 2:
                audio_a = audio_a[np.newaxis, :, :]
            a_t = torch.from_numpy(audio_a).float().to(self.device)
        else:
            a_t = audio_a.to(self.device)
            if a_t.ndim == 1:
                a_t = a_t.unsqueeze(0).unsqueeze(0)
            elif a_t.ndim == 2:
                a_t = a_t.unsqueeze(0)
                
        if isinstance(audio_b, np.ndarray):
            if audio_b.ndim == 1:
                audio_b = audio_b[np.newaxis, np.newaxis, :]
            elif audio_b.ndim == 2:
                audio_b = audio_b[np.newaxis, :, :]
            b_t = torch.from_numpy(audio_b).float().to(self.device)
        else:
            b_t = audio_b.to(self.device)
            if b_t.ndim == 1:
                b_t = b_t.unsqueeze(0).unsqueeze(0)
            elif b_t.ndim == 2:
                b_t = b_t.unsqueeze(0)

        t0 = time.perf_counter()
        delta, (logit_a, logit_b), _ = self.runner(eeg_t, a_t, b_t)
        t1 = time.perf_counter()
        
        compute_ms = (t1 - t0) * 1000.0
        return {
            "delta": float(delta.item()),
            "logit_a": float(logit_a.item()),
            "logit_b": float(logit_b.item()),
            "compute_ms": compute_ms,
        }

    def benchmark(self, n_iters: int = 100, window_samples: int = 320) -> dict:
        """Benchmarks inference latency over n_iters runs."""
        dummy_eeg = np.random.randn(1, self.in_channels, window_samples).astype(np.float32)
        dummy_a = np.random.randn(1, 1, window_samples).astype(np.float32)
        dummy_b = np.random.randn(1, 1, window_samples).astype(np.float32)
        
        latencies = []
        for _ in range(n_iters):
            res = self.predict(dummy_eeg, dummy_a, dummy_b)
            latencies.append(res["compute_ms"])
            
        latencies = np.array(latencies)
        return {
            "mode": self.mode,
            "device": str(self.device),
            "n_iters": n_iters,
            "mean_ms": float(np.mean(latencies)),
            "std_ms": float(np.std(latencies)),
            "p50_ms": float(np.median(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
        }
