"""
Single-Channel Neural Speech Separator for Phone & Voice-Memo Recordings.

Uses pre-trained Conv-TasNet (Convolutional Time-domain Audio Separation Network)
to separate two overlapping speakers from a single-microphone recording:
  1. Ingests any audio format (.wav, .mp3, .m4a, .flac) or raw numpy array.
  2. Downmixes stereo/multi-channel to mono.
  3. Resamples to 8 kHz for Conv-TasNet inference.
  4. Separates mixture into Speaker 1 and Speaker 2 waveforms.
  5. Resamples back to 16 kHz and extracts causal Gammatone envelopes at 64 Hz for CA-TCN AAD.
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wavfile
from scipy import signal
import torch
import torchaudio
from torchaudio.pipelines import CONVTASNET_BASE_LIBRI2MIX

from src.audio.causal_gammatone import StreamingCausalAudioGammatoneExtractor


class SingleChannelNeuralSeparator:
    """
    Separates two competing talkers from a single microphone audio track (e.g. phone recording).
    
    Parameters:
        device: 'cuda' if GPU available, else 'cpu'.
        target_fs: Output sampling rate for separated waveforms (default: 16000 Hz).
    """
    def __init__(self, device: Optional[str] = None, target_fs: int = 16000):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.target_fs = int(target_fs)
        self.bundle = CONVTASNET_BASE_LIBRI2MIX
        self.model_fs = self.bundle.sample_rate  # 8000 Hz

        # Load pre-trained Conv-TasNet model
        self.model = self.bundle.get_model().to(self.device).eval()

    def _resample(self, audio: np.ndarray, orig_fs: int, target_fs: int) -> np.ndarray:
        """Resamples 1D audio using polyphase filtering."""
        if orig_fs == target_fs:
            return audio.astype(np.float32)
        gcd = np.gcd(orig_fs, target_fs)
        up = target_fs // gcd
        down = orig_fs // gcd
        return signal.resample_poly(audio, up, down).astype(np.float32)

    def separate_waveform(self, audio: np.ndarray, orig_fs: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Separates a 1D or 2D single-microphone audio recording into two speaker streams.
        
        Parameters:
            audio: 1D numpy array [N] or 2D array [channels, N] / [N, channels].
            orig_fs: Input sampling rate in Hz.
            
        Returns:
            speaker_1: 1D numpy array at target_fs (16 kHz).
            speaker_2: 1D numpy array at target_fs (16 kHz).
        """
        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim > 1:
            # Downmix stereo to mono
            if arr.shape[0] <= 4 and arr.shape[1] > arr.shape[0]:
                arr = np.mean(arr, axis=0)
            else:
                arr = np.mean(arr, axis=-1)
        arr = arr.reshape(-1)

        # Remove DC offset & normalize
        arr = arr - np.mean(arr)
        peak = np.max(np.abs(arr)) + 1e-8
        arr = arr / peak

        # Resample to 8 kHz for Conv-TasNet
        arr_8k = self._resample(arr, orig_fs, self.model_fs)

        # Prepare tensor: (batch=1, channel=1, time)
        mix_tensor = torch.from_numpy(arr_8k).view(1, 1, -1).to(self.device)

        with torch.no_grad():
            # Conv-TasNet forward pass -> (1, 2, time)
            separated_tensor = self.model(mix_tensor)

        # Extract two speaker channels
        s1_8k = separated_tensor[0, 0, :].cpu().numpy()
        s2_8k = separated_tensor[0, 1, :].cpu().numpy()

        # Resample back to target_fs (16 kHz)
        s1_out = self._resample(s1_8k, self.model_fs, self.target_fs)
        s2_out = self._resample(s2_8k, self.model_fs, self.target_fs)

        # Peak normalize to -1.0 dBFS ceiling
        max_s1 = np.max(np.abs(s1_out)) + 1e-8
        max_s2 = np.max(np.abs(s2_out)) + 1e-8
        s1_out = (s1_out / max_s1) * 0.89
        s2_out = (s2_out / max_s2) * 0.89

        return s1_out, s2_out

    def separate_file(
        self,
        input_audio_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Path, Path]:
        """
        Loads an audio file (e.g. phone recording), separates it, and writes output WAVs.
        
        Parameters:
            input_audio_path: Path to phone recording (.wav, .mp3, etc.).
            output_dir: Directory where separated tracks will be saved.
            
        Returns:
            s1_out: 1D numpy array of Speaker 1.
            s2_out: 1D numpy array of Speaker 2.
            path_s1: Path to saved Speaker 1 WAV file.
            path_s2: Path to saved Speaker 2 WAV file.
        """
        input_path = Path(input_audio_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input audio file not found: {input_path}")

        # Load audio via torchaudio or scipy
        try:
            waveform, orig_fs = torchaudio.load(str(input_path))
            audio_np = waveform.numpy()
        except Exception:
            orig_fs, audio_np = wavfile.read(str(input_path))
            if audio_np.dtype == np.int16:
                audio_np = audio_np.astype(np.float32) / 32768.0

        s1_out, s2_out = self.separate_waveform(audio_np, orig_fs)

        if output_dir is None:
            output_dir = input_path.parent / f"{input_path.stem}_separated"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        path_s1 = output_dir / f"{input_path.stem}_speaker_1.wav"
        path_s2 = output_dir / f"{input_path.stem}_speaker_2.wav"

        # Save 16-bit PCM WAVs
        wavfile.write(str(path_s1), self.target_fs, (s1_out * 32767.0).astype(np.int16))
        wavfile.write(str(path_s2), self.target_fs, (s2_out * 32767.0).astype(np.int16))

        return s1_out, s2_out, path_s1, path_s2

    def extract_neural_envelopes(
        self,
        speaker_1_audio: np.ndarray,
        speaker_2_audio: np.ndarray,
        envelope_fs: float = 64.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts 64 Hz causal Gammatone envelopes from both separated streams for CA-TCN AAD.
        
        Returns:
            env_1: 1D broadband envelope of Speaker 1 at 64 Hz.
            env_2: 1D broadband envelope of Speaker 2 at 64 Hz.
        """
        ext1 = StreamingCausalAudioGammatoneExtractor(audio_fs=self.target_fs, target_fs=envelope_fs)
        ext2 = StreamingCausalAudioGammatoneExtractor(audio_fs=self.target_fs, target_fs=envelope_fs)

        chunk_size = 500  # 31.25 ms chunks @ 16 kHz
        env1_chunks = []
        env2_chunks = []

        min_len = min(len(speaker_1_audio), len(speaker_2_audio))
        for i in range(0, min_len, chunk_size):
            c1 = speaker_1_audio[i:i + chunk_size]
            c2 = speaker_2_audio[i:i + chunk_size]
            e1 = ext1.process_audio_chunk(c1)
            e2 = ext2.process_audio_chunk(c2)
            if len(e1) > 0:
                env1_chunks.append(e1)
            if len(e2) > 0:
                env2_chunks.append(e2)

        env1 = np.concatenate(env1_chunks) if env1_chunks else np.zeros(0, dtype=np.float32)
        env2 = np.concatenate(env2_chunks) if env2_chunks else np.zeros(0, dtype=np.float32)

        return env1, env2
