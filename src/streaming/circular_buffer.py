import numpy as np
import threading

class SynchronizedRingBuffer:
    """
    Fixed-size, thread-safe circular ring buffer for time-synchronized EEG and dual audio streams.
    
    Attributes:
        capacity (int): Total buffer length in samples (e.g. 5.0 s * 64 Hz = 320 samples).
        n_eeg_channels (int): Number of EEG channels (e.g. 8).
    """
    def __init__(self, capacity: int = 320, n_eeg_channels: int = 8):
        self.capacity = capacity
        self.n_eeg_channels = n_eeg_channels
        
        # Pre-allocated circular storage: [channels, capacity]
        self._eeg_buf = np.zeros((n_eeg_channels, capacity), dtype=np.float32)
        self._audio_a_buf = np.zeros((1, capacity), dtype=np.float32)
        self._audio_b_buf = np.zeros((1, capacity), dtype=np.float32)
        
        self._head = 0  # Points to the next write index
        self._total_samples = 0
        self._lock = threading.Lock()

    def reset(self):
        """Clears all buffered data and resets pointers."""
        with self._lock:
            self._eeg_buf.fill(0.0)
            self._audio_a_buf.fill(0.0)
            self._audio_b_buf.fill(0.0)
            self._head = 0
            self._total_samples = 0

    def is_ready(self) -> bool:
        """Returns True once the buffer has accumulated at least `capacity` samples."""
        return self._total_samples >= self.capacity

    def push(self, eeg: np.ndarray, audio_a: np.ndarray, audio_b: np.ndarray):
        """
        Pushes new time-synchronized samples into the circular buffer.
        
        Parameters:
            eeg: [n_samples, n_eeg_channels] or [n_eeg_channels, n_samples]
            audio_a: [n_samples] or [1, n_samples]
            audio_b: [n_samples] or [1, n_samples]
        """
        # Ensure correct shapes
        eeg = np.asarray(eeg, dtype=np.float32)
        audio_a = np.asarray(audio_a, dtype=np.float32).reshape(1, -1)
        audio_b = np.asarray(audio_b, dtype=np.float32).reshape(1, -1)
        
        if eeg.shape[0] != self.n_eeg_channels and eeg.shape[1] == self.n_eeg_channels:
            eeg = eeg.T  # Transpose to [n_eeg_channels, n_samples]
            
        n_samples = eeg.shape[1]
        assert audio_a.shape[1] == n_samples, f"Audio A samples ({audio_a.shape[1]}) != EEG ({n_samples})"
        assert audio_b.shape[1] == n_samples, f"Audio B samples ({audio_b.shape[1]}) != EEG ({n_samples})"
        
        if n_samples == 0:
            return

        with self._lock:
            if n_samples >= self.capacity:
                # If incoming chunk is larger than buffer capacity, take the last `capacity` samples
                self._eeg_buf[:] = eeg[:, -self.capacity:]
                self._audio_a_buf[:] = audio_a[:, -self.capacity:]
                self._audio_b_buf[:] = audio_b[:, -self.capacity:]
                self._head = 0
            else:
                tail_space = self.capacity - self._head
                if n_samples <= tail_space:
                    self._eeg_buf[:, self._head:self._head + n_samples] = eeg
                    self._audio_a_buf[:, self._head:self._head + n_samples] = audio_a
                    self._audio_b_buf[:, self._head:self._head + n_samples] = audio_b
                    self._head = (self._head + n_samples) % self.capacity
                else:
                    # Wraparound write
                    part1 = tail_space
                    part2 = n_samples - tail_space
                    
                    self._eeg_buf[:, self._head:] = eeg[:, :part1]
                    self._audio_a_buf[:, self._head:] = audio_a[:, :part1]
                    self._audio_b_buf[:, self._head:] = audio_b[:, :part1]
                    
                    self._eeg_buf[:, :part2] = eeg[:, part1:]
                    self._audio_a_buf[:, :part2] = audio_a[:, part1:]
                    self._audio_b_buf[:, :part2] = audio_b[:, part1:]
                    
                    self._head = part2
                    
            self._total_samples += n_samples

    def snapshot(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns a time-ordered snapshot of the circular buffer.
        
        Returns:
            (eeg_window, audio_a_window, audio_b_window)
            Shapes: [1, n_eeg_channels, capacity], [1, 1, capacity], [1, 1, capacity]
            Chronologically ordered from oldest sample (index 0) to newest sample (index -1).
        """
        with self._lock:
            if self._total_samples < self.capacity:
                # Still filling up: zero-pad on left (oldest past)
                valid = min(self._total_samples, self._head)
                eeg_out = np.zeros((1, self.n_eeg_channels, self.capacity), dtype=np.float32)
                a_out = np.zeros((1, 1, self.capacity), dtype=np.float32)
                b_out = np.zeros((1, 1, self.capacity), dtype=np.float32)
                
                if valid > 0:
                    eeg_out[0, :, -valid:] = self._eeg_buf[:, :valid]
                    a_out[0, 0, -valid:] = self._audio_a_buf[0, :valid]
                    b_out[0, 0, -valid:] = self._audio_b_buf[0, :valid]
                return eeg_out, a_out, b_out
                
            # Buffer is full, unroll from oldest (_head) to newest
            if self._head == 0:
                eeg_ordered = self._eeg_buf.copy()
                a_ordered = self._audio_a_buf.copy()
                b_ordered = self._audio_b_buf.copy()
            else:
                eeg_ordered = np.concatenate([self._eeg_buf[:, self._head:], self._eeg_buf[:, :self._head]], axis=1)
                a_ordered = np.concatenate([self._audio_a_buf[:, self._head:], self._audio_a_buf[:, :self._head]], axis=1)
                b_ordered = np.concatenate([self._audio_b_buf[:, self._head:], self._audio_b_buf[:, :self._head]], axis=1)
                
            return (
                eeg_ordered[np.newaxis, :, :],
                a_ordered[np.newaxis, :, :],
                b_ordered[np.newaxis, :, :]
            )
