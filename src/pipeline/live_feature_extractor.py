import numpy as np
from scipy.signal import butter, sosfilt, hilbert

def get_log_spaced_edges(fmin, fmax, num_bands):
    return np.logspace(np.log10(fmin), np.log10(fmax), num=num_bands + 1)

class LiveFeatureExtractor:
    """
    Stateful feature extractor designed for real-time audio chunk processing.
    Maintains filter states across chunks to ensure continuous processing without edge artifacts.
    """
    def __init__(self, num_bands=8, fmin=100, fmax=8000, fs=48000, target_fs=64):
        self.num_bands = num_bands
        self.fs = fs
        self.target_fs = target_fs
        
        edges = get_log_spaced_edges(fmin, fmax, num_bands)
        nyq = 0.5 * fs
        
        self.bp_sos = []
        self.bp_zi = []
        
        # Pre-compute bandpass SOS filters and initialize state (zi)
        for i in range(num_bands):
            low = edges[i]
            high = edges[i+1]
            sos = butter(4, [low / nyq, min(high / nyq, 0.99)], btype='band', output='sos')
            self.bp_sos.append(sos)
            
            # Initial state of the filter
            from scipy.signal import sosfilt_zi
            zi = sosfilt_zi(sos)
            self.bp_zi.append(zi * 0.0) # initialize with zeros
            
        # Low pass filter for envelope at 8 Hz
        self.lp_sos = butter(4, 8.0 / nyq, btype='low', output='sos')
        from scipy.signal import sosfilt_zi
        self.lp_zi = [sosfilt_zi(self.lp_sos) * 0.0 for _ in range(num_bands)]
        
        # Buffer for resampling (to handle downsampling from 48kHz to 64Hz)
        self.downsample_factor = fs // target_fs
        self.resample_buffer = [np.array([]) for _ in range(num_bands)]
        
    def process_chunk(self, audio_chunk):
        """
        Process a new incoming chunk of audio data (e.g. 50ms = 2400 samples at 48kHz).
        Returns a (8, N_downsampled) feature matrix, where N_downsampled may be 0 if the chunk is too small.
        """
        # Ensure mono
        if len(audio_chunk.shape) > 1:
            audio_chunk = np.mean(audio_chunk, axis=1)
            
        bands = []
        for i in range(self.num_bands):
            # 1. Stateful Bandpass Filter
            filtered, self.bp_zi[i] = sosfilt(self.bp_sos[i], audio_chunk, zi=self.bp_zi[i])
            
            # 2. Hilbert Transform (Note: true hilbert needs future context, 
            # in real-time we often approximate this or use an overlapping buffer.
            # Here we apply it on the chunk, which is a known limitation of purely causal envelope extraction)
            analytic = hilbert(filtered)
            envelope = np.abs(analytic)
            
            # 3. Stateful Lowpass Filter (8 Hz)
            env_lp, self.lp_zi[i] = sosfilt(self.lp_sos, envelope, zi=self.lp_zi[i])
            
            # 4. Decimation (Integer downsampling for real-time simplicity)
            # Append to buffer
            self.resample_buffer[i] = np.concatenate((self.resample_buffer[i], env_lp))
            
            # Extract completed 64Hz samples
            num_samples = len(self.resample_buffer[i]) // self.downsample_factor
            if num_samples > 0:
                downsampled = self.resample_buffer[i][:num_samples * self.downsample_factor:self.downsample_factor]
                # Remove consumed samples from buffer
                self.resample_buffer[i] = self.resample_buffer[i][num_samples * self.downsample_factor:]
                bands.append(downsampled)
            else:
                bands.append(np.array([]))
                
        # Return matrix of shape (8, N), where N is number of new 64Hz samples generated
        # If no samples were generated (e.g. very small chunk), return empty
        if len(bands[0]) == 0:
            return np.zeros((self.num_bands, 0))
            
        return np.vstack(bands)

    def reset_state(self):
        """Reset the internal state of all filters and buffers."""
        from scipy.signal import sosfilt_zi
        self.bp_zi = [sosfilt_zi(sos) * 0.0 for sos in self.bp_sos]
        self.lp_zi = [sosfilt_zi(self.lp_sos) * 0.0 for _ in range(self.num_bands)]
        self.resample_buffer = [np.array([]) for _ in range(self.num_bands)]
