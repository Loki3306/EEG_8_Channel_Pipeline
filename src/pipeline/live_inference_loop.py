import os
import sys
import numpy as np
import torch
import time
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(REPO_ROOT))

from src.pipeline.live_streamer import LiveStreamSimulator
from models.aad_conformer import AADConformer
from decision_engine.decision_policy_engine import DecisionPolicyEngine
from data.extract_subband_envelopes import extract_subband_envelopes
import tempfile
from scipy.io import wavfile

class SlidingBuffer:
    def __init__(self, size, shape_template):
        """
        shape_template is the shape of a single sample or the total shape.
        For Audio: size=96000 (2s at 48kHz), shape_template=(1,) -> buffer shape (96000,)
        For EEG: size=128 (2s at 64Hz), shape_template=(64,) -> buffer shape (64, 128)
        """
        self.size = size
        if isinstance(shape_template, int):
            self.buffer = np.zeros(size)
        else:
            self.buffer = np.zeros((shape_template[0], size))
            
    def append(self, chunk):
        """
        Append a new chunk to the end of the buffer and discard the oldest data.
        chunk: shape (N,) or (64, N)
        """
        chunk_len = len(chunk) if len(chunk.shape) == 1 else chunk.shape[1]
        if chunk_len >= self.size:
            # Chunk is larger than buffer, just take the end
            if len(chunk.shape) == 1:
                self.buffer = chunk[-self.size:]
            else:
                self.buffer = chunk[:, -self.size:]
        else:
            if len(chunk.shape) == 1:
                self.buffer[:-chunk_len] = self.buffer[chunk_len:]
                self.buffer[-chunk_len:] = chunk
            else:
                self.buffer[:, :-chunk_len] = self.buffer[:, chunk_len:]
                self.buffer[:, -chunk_len:] = chunk
                
    def get_data(self):
        return self.buffer

class LiveInferenceLoop:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = AADConformer(
            eeg_channels=64, 
            audio_channels=1, 
            embed_dim=256, 
            depth=6, 
            heads=8
        ).to(device)
        
        state_dict = torch.load(model_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # Adaptive Threshold Engine from Phase 16B
        self.engine = DecisionPolicyEngine(confidence_threshold=0.85, required_confirmations=3)
        
        # 2-second buffers
        self.audio_fs = 48000
        self.eeg_fs = 64
        self.window_sec = 2.0
        
        self.eeg_buffer = SlidingBuffer(int(self.window_sec * self.eeg_fs), (64,))
        self.audio_a_buffer = SlidingBuffer(int(self.window_sec * self.audio_fs), 1)
        self.audio_b_buffer = SlidingBuffer(int(self.window_sec * self.audio_fs), 1)
        
    def _extract_envelopes(self, audio_array):
        # We write to temp wav to use the exact offline extraction function
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            wav_path = f.name
            
        audio_int16 = np.int16(audio_array / (np.max(np.abs(audio_array)) + 1e-8) * 32767)
        wavfile.write(wav_path, self.audio_fs, audio_int16)
        
        env = extract_subband_envelopes(wav_path, target_fs=self.eeg_fs)
        os.remove(wav_path)
        return env

    def process_stream(self, streamer):
        print("Starting real-time inference loop...")
        
        step = 0
        for e_chunk, a_chunk, b_chunk in streamer:
            # 1. Update sliding buffers
            self.eeg_buffer.append(e_chunk)
            self.audio_a_buffer.append(a_chunk)
            self.audio_b_buffer.append(b_chunk)
            
            # Wait until buffer is full before predicting
            step += 1
            # 50ms chunk = 20 chunks per second. 2 seconds = 40 chunks
            if step < 40:
                continue
                
            # 2. Extract features over the 2-second window
            eeg_win = self.eeg_buffer.get_data()
            audio_a_win = self.audio_a_buffer.get_data()
            audio_b_win = self.audio_b_buffer.get_data()
            
            env_a = self._extract_envelopes(audio_a_win)
            env_b = self._extract_envelopes(audio_b_win)
            
            # 3. Model Inference
            e_tensor = torch.tensor(eeg_win, dtype=torch.float32).unsqueeze(0).to(self.device)
            a_tensor = torch.tensor(env_a, dtype=torch.float32).unsqueeze(0).to(self.device)
            b_tensor = torch.tensor(env_b, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Normalize Audio as in training
            a_tensor = (a_tensor - a_tensor.mean(dim=-1, keepdim=True)) / (a_tensor.std(dim=-1, keepdim=True) + 1e-8)
            b_tensor = (b_tensor - b_tensor.mean(dim=-1, keepdim=True)) / (b_tensor.std(dim=-1, keepdim=True) + 1e-8)
            
            with torch.no_grad():
                pred = self.model(e_tensor)
                
                # Pearson correlation
                pred_c = pred - pred.mean(dim=-1, keepdim=True)
                a_c = a_tensor - a_tensor.mean(dim=-1, keepdim=True)
                b_c = b_tensor - b_tensor.mean(dim=-1, keepdim=True)
                
                ca = (pred_c * a_c).sum(dim=-1) / (torch.sqrt((pred_c**2).sum(dim=-1) * (a_c**2).sum(dim=-1)) + 1e-8)
                cb = (pred_c * b_c).sum(dim=-1) / (torch.sqrt((pred_c**2).sum(dim=-1) * (b_c**2).sum(dim=-1)) + 1e-8)
                
                ca = ca.item()
                cb = cb.item()
                
            # Calibrate using Phase 13 Platt scaling constants (Global Absolute Margin)
            # Platt a=-0.0524, b=-0.0250 (from S3 global absolute margin, roughly average)
            # Actually for product, we will just use softmax of correlations for now
            # or simply prob = ca > cb
            margin = ca - cb
            prob = 1.0 / (1.0 + np.exp(-(-0.4991 * margin - 0.1519))) # Example Platt constants from S7 global
            
            # 4. Policy Engine
            dec = self.engine.update(prob, margin)
            
            print(f"Window {step} | ca:{ca:.3f} cb:{cb:.3f} prob:{prob:.3f} | State:{dec['state']} Action:{dec['action']}")

if __name__ == "__main__":
    # Test run
    fs = 48000
    t = np.linspace(0, 10, 10 * fs, endpoint=False)
    audio = np.sin(2 * np.pi * 1000 * t)
    eeg = np.zeros((64, 10 * 64))
    
    streamer = LiveStreamSimulator(audio_fs=fs, eeg_fs=64, chunk_size_ms=50).stream_trial(eeg, audio, audio)
    
    # Needs a real model checkpoint. For test we mock it.
    print("Please provide a valid AADConformer checkpoint.")
