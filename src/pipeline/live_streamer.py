import numpy as np
import time

class LiveStreamSimulator:
    """
    Simulates a continuous hardware data stream.
    Takes full offline arrays (e.g. from KUL dataset) and yields them in real-time sized chunks.
    """
    def __init__(self, audio_fs=48000, eeg_fs=64, chunk_size_ms=50):
        self.audio_fs = audio_fs
        self.eeg_fs = eeg_fs
        self.chunk_size_ms = chunk_size_ms
        
        self.audio_chunk_size = int(audio_fs * (chunk_size_ms / 1000.0))
        self.eeg_chunk_size = int(eeg_fs * (chunk_size_ms / 1000.0))
        
    def stream_trial(self, eeg_data, audio_a_data, audio_b_data, real_time_delay=False):
        """
        Generator that yields synchronous chunks of EEG, Audio A, and Audio B.
        - eeg_data: (64, N_eeg_samples)
        - audio_a_data: (N_audio_samples,)
        - audio_b_data: (N_audio_samples,)
        
        Yields:
        (eeg_chunk, audio_a_chunk, audio_b_chunk)
        """
        # Ensure Audio is mono (shape: N_audio_samples)
        if len(audio_a_data.shape) > 1: audio_a_data = np.mean(audio_a_data, axis=1)
        if len(audio_b_data.shape) > 1: audio_b_data = np.mean(audio_b_data, axis=1)
        
        total_audio_samples = len(audio_a_data)
        current_audio_idx = 0
        current_eeg_idx = 0
        
        while current_audio_idx + self.audio_chunk_size <= total_audio_samples:
            if real_time_delay:
                time.sleep(self.chunk_size_ms / 1000.0)
                
            a_chunk = audio_a_data[current_audio_idx : current_audio_idx + self.audio_chunk_size]
            b_chunk = audio_b_data[current_audio_idx : current_audio_idx + self.audio_chunk_size]
            
            # Note: EEG is shape (Channels, Time)
            e_chunk = eeg_data[:, current_eeg_idx : current_eeg_idx + self.eeg_chunk_size]
            
            yield e_chunk, a_chunk, b_chunk
            
            current_audio_idx += self.audio_chunk_size
            current_eeg_idx += self.eeg_chunk_size
            
        # Optional: yield remainder
        if current_audio_idx < total_audio_samples:
            a_chunk = audio_a_data[current_audio_idx:]
            b_chunk = audio_b_data[current_audio_idx:]
            e_chunk = eeg_data[:, current_eeg_idx:]
            yield e_chunk, a_chunk, b_chunk
