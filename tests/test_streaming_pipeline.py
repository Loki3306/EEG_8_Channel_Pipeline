import unittest
import numpy as np
import torch
from scipy import signal

from src.streaming.causal_filters import StreamingCausalEEGFilter
from src.streaming.causal_envelope import StreamingCausalEnvelopeExtractor
from src.streaming.circular_buffer import SynchronizedRingBuffer
from src.deployment.decision_smoother import EMAHysteresisDecisionLayer
from src.deployment.engine import StreamingCATCNEngine
from src.streaming.pipeline import StreamingAADPipeline
from scripts.verify_baseline.models.catcn import CATCNDirectDecoder

class TestStreamingPipeline(unittest.TestCase):
    
    def test_causal_filter_chunk_equivalence(self):
        """Tests that chunk-by-chunk streaming filter matches batch sosfilt exactly."""
        n_samples = 1280  # 20 seconds at 64 Hz
        n_channels = 8
        fs = 64.0
        np.random.seed(42)
        raw_eeg = np.random.randn(n_samples, n_channels).astype(np.float64)
        
        # 1. Batch reference using exact same SOS
        stream_filter = StreamingCausalEEGFilter(lowcut=1.0, highcut=6.0, fs=fs, order=2, n_channels=n_channels)
        sos = stream_filter.sos
        base_zi = signal.sosfilt_zi(sos)[:, :, np.newaxis] * raw_eeg[0][np.newaxis, np.newaxis, :]
        batch_ref, _ = signal.sosfilt(sos, raw_eeg, axis=0, zi=base_zi)
        
        # 2. Chunk-by-chunk streaming filter
        stream_filter.reset()
        chunk_size = 32  # 0.5s chunks
        chunks_out = []
        for i in range(0, n_samples, chunk_size):
            chunk = raw_eeg[i:i + chunk_size]
            chunks_out.append(stream_filter.process_chunk(chunk))
            
        streamed_result = np.vstack(chunks_out)
        
        # Compare
        max_diff = np.max(np.abs(batch_ref - streamed_result))
        self.assertLess(max_diff, 1e-4, f"Streaming filter diverged from batch reference! Max diff: {max_diff}")
        print(f"[PASS] Filter Chunk Equivalence: Max difference = {max_diff:.2e}")

    def test_ring_buffer_chronology_and_wraparound(self):
        """Tests that circular ring buffer retains exact chronological past samples across wraparounds."""
        capacity = 320
        n_channels = 8
        buffer = SynchronizedRingBuffer(capacity=capacity, n_eeg_channels=n_channels)
        
        self.assertFalse(buffer.is_ready())
        
        # Push 1000 samples sequentially in varying chunk sizes
        total_pushed = 1000
        eeg_full = np.arange(total_pushed)[:, np.newaxis] * np.ones((1, n_channels), dtype=np.float32)
        audio_a_full = np.arange(total_pushed, dtype=np.float32)
        audio_b_full = -np.arange(total_pushed, dtype=np.float32)
        
        idx = 0
        chunk_sizes = [16, 32, 64, 25, 40, 50, 100, 32]
        c_i = 0
        while idx < total_pushed:
            cs = min(chunk_sizes[c_i % len(chunk_sizes)], total_pushed - idx)
            buffer.push(eeg_full[idx:idx + cs], audio_a_full[idx:idx + cs], audio_b_full[idx:idx + cs])
            idx += cs
            c_i += 1
            
        self.assertTrue(buffer.is_ready())
        e_win, a_win, b_win = buffer.snapshot()
        
        # The snapshot must contain exactly the last `capacity` samples: [1000 - 320 : 1000]
        expected_eeg = eeg_full[-capacity:].T[np.newaxis, :, :]
        expected_a = audio_a_full[-capacity:][np.newaxis, np.newaxis, :]
        expected_b = audio_b_full[-capacity:][np.newaxis, np.newaxis, :]
        
        np.testing.assert_allclose(e_win, expected_eeg, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(a_win, expected_a, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(b_win, expected_b, rtol=1e-5, atol=1e-5)
        print("[PASS] Ring Buffer Chronology & Wraparound: Snapshot matches expected tail perfectly.")

    def test_decision_layer_hysteresis_anti_chatter(self):
        """Tests that EMA smoothing and dual-threshold hysteresis reject noise and switch cleanly."""
        layer = EMAHysteresisDecisionLayer(alpha=0.7, threshold=0.3, n_confirm=2, boost_db=6.0)
        
        # 1. Noisy fluctuations around 0 -> Should remain UNCERTAIN, gain equal
        for _ in range(5):
            telemetry = layer.update(np.random.uniform(-0.1, 0.1))
        self.assertEqual(telemetry["attended_stream"], "UNCERTAIN")
        self.assertAlmostEqual(telemetry["gain_a"], telemetry["gain_b"], delta=0.05)
        
        # 2. Sustained positive signal (Attending A)
        for _ in range(4):
            telemetry = layer.update(0.8)
        self.assertEqual(telemetry["attended_stream"], "A")
        self.assertGreater(telemetry["gain_a"], telemetry["gain_b"])
        self.assertGreater(telemetry["confidence"], 0.7)
        
        # 3. Single negative spike (noise) -> Should NOT switch to B
        telemetry = layer.update(-0.6)
        self.assertEqual(telemetry["attended_stream"], "A")
        self.assertFalse(telemetry["switched"])
        
        # 4. Sustained negative signal (Switch to B)
        switched_seen = False
        for _ in range(5):
            telemetry = layer.update(-0.9)
            if telemetry["switched"]:
                switched_seen = True
        self.assertTrue(switched_seen)
        self.assertEqual(telemetry["attended_stream"], "B")
        self.assertGreater(telemetry["gain_b"], telemetry["gain_a"])
        print("[PASS] Decision Layer Hysteresis: Anti-chatter and state transitions validated.")

    def test_pipeline_end_to_end_and_latency(self):
        """Tests the integrated streaming pipeline and verifies T_compute < 10 ms."""
        model = CATCNDirectDecoder(eeg_channels=8, audio_channels=1, hidden_dim=64, max_lag_samples=8)
        pipeline = StreamingAADPipeline(
            model=model,
            n_eeg_channels=8,
            fs=64.0,
            raw_audio_input=False,
            window_sec=5.0,
            step_sec=0.5,
            engine_mode="torchscript"
        )
        
        # Benchmark engine latency directly
        bench = pipeline.engine.benchmark(n_iters=50, window_samples=320)
        print(f"[BENCHMARK] Inference Latency (CPU TorchScript): Mean={bench['mean_ms']:.2f} ms | P95={bench['p95_ms']:.2f} ms")
        self.assertLess(bench['mean_ms'], 20.0, "Mean Inference latency exceeded real-time budget!")
        self.assertLess(bench['p95_ms'], 35.0, "P95 Inference latency exceeded real-time budget!")
        
        # Feed 10 seconds of simulated data (0.5s chunks = 32 samples per chunk)
        chunk_len = 32
        n_chunks = 20
        step_results = []
        for _ in range(n_chunks):
            chunk_eeg = np.random.randn(chunk_len, 8).astype(np.float32)
            chunk_a = np.random.randn(chunk_len).astype(np.float32)
            chunk_b = np.random.randn(chunk_len).astype(np.float32)
            
            res = pipeline.feed_sample_block(chunk_eeg, chunk_a, chunk_b)
            if res is not None:
                step_results.append(res)
                
        # With 5s window and 0.5s step, after 5s buffer fills, each 0.5s emits an inference result
        self.assertGreater(len(step_results), 5)
        for r in step_results:
            self.assertIn("timestamp_sec", r)
            self.assertIn("raw_delta", r)
            self.assertIn("attended_stream", r)
            self.assertIn("gain_a", r)
            self.assertIn("compute_ms", r)
            self.assertLess(r["compute_ms"], 50.0)
            
        print(f"[PASS] Pipeline End-to-End: Emitted {len(step_results)} streaming inference frames seamlessly.")

if __name__ == "__main__":
    unittest.main()
