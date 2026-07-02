import os
import json
import numpy as np
import torch
from pathlib import Path

class DatasetAdapter:
    """Abstract base class for loading dataset-specific segments."""
    def get_segment(self, subject, trial, start_sec, duration_sec):
        """
        Returns:
        eeg: (Channels, Samples)
        audio_a: (Samples,)
        audio_b: (Samples,)
        target_label: int (1 for Audio A, 0 for Audio B)
        """
        raise NotImplementedError

class KULAdapter(DatasetAdapter):
    def __init__(self, cache_dir="/kaggle/input/datasets/lokeshgile/kul-processed/data/processed_kul", fs=64):
        self.cache_dir = Path(cache_dir)
        self.fs = fs
        self.loaded_data = None
        
    def _load_data(self):
        if self.loaded_data is None:
            # We must load using KULCachedLoader. 
            # We can import it dynamically to avoid circular dependencies if needed, 
            # but since it's just a class, we import it here.
            import sys
            # Find repo root
            root = Path(__file__).resolve().parent.parent.parent
            if str(root) not in sys.path:
                sys.path.append(str(root))
            from data.kul_cached_dataset import KULCachedLoader
            loader = KULCachedLoader(str(self.cache_dir))
            self.loaded_data = loader.load_all()
            
    def get_segment(self, subject, trial, start_sec, duration_sec):
        self._load_data()
        
        if subject not in self.loaded_data:
            raise ValueError(f"Subject {subject} not found in KUL dataset.")
            
        # Trial is 1-indexed in JSON, but typically datasets might be 0-indexed or 1-indexed.
        # KULCachedLoader gives a list of trials. We assume trial 1 -> index 0.
        trial_idx = trial - 1
        trials = self.loaded_data[subject]
        
        if trial_idx < 0 or trial_idx >= len(trials):
            raise ValueError(f"Trial {trial} not found for Subject {subject}.")
            
        trial_data = trials[trial_idx]
        eeg = trial_data['eeg']
        ya = trial_data['audio_a']
        yb = trial_data['audio_b']
        # In KUL cache, audio_a is always the attended stimulus, so ground truth is 1.
        label = 1
        
        start_samp = int(start_sec * self.fs)
        if duration_sec is None:
            end_samp = eeg.shape[1]
        else:
            end_samp = start_samp + int(duration_sec * self.fs)
            
        eeg_seg = eeg[:, start_samp:end_samp]
        ya_seg = ya[:, start_samp:end_samp] if len(ya.shape) > 1 else ya[start_samp:end_samp]
        yb_seg = yb[:, start_samp:end_samp] if len(yb.shape) > 1 else yb[start_samp:end_samp]
        
        # Ensure audio is 1D
        if len(ya_seg.shape) > 1: ya_seg = ya_seg.mean(axis=0)
        if len(yb_seg.shape) > 1: yb_seg = yb_seg.mean(axis=0)
        
        # Convert tensors to numpy if necessary
        if isinstance(eeg_seg, torch.Tensor): eeg_seg = eeg_seg.numpy()
        if isinstance(ya_seg, torch.Tensor): ya_seg = ya_seg.numpy()
        if isinstance(yb_seg, torch.Tensor): yb_seg = yb_seg.numpy()
        
        return eeg_seg, ya_seg, yb_seg, label


class ContinuousSessionGenerator:
    def __init__(self, adapters=None):
        self.adapters = adapters or {}
        self.fs = 64
        self.window_sec = 2.0
        self.hop_sec = 0.05
        
    def _parse_scenario(self, json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
            
    def generate_stream(self, json_path):
        """
        Parses the JSON scenario, fetches segments, concatenates them, 
        and yields continuous windows.
        """
        scenario = self._parse_scenario(json_path)
        scenario_name = scenario['scenario_name']
        
        eeg_list, ya_list, yb_list = [], [], []
        window_metadata = [] # Tracks ground truth and scene per sample
        
        # 1. Fetch and Stitch
        for scene in scenario['scenes']:
            dataset = scene['dataset']
            subject = scene['subject']
            trial = scene['trial']
            start_sec = scene.get('start_sec', 0)
            dur_sec = scene.get('duration_seconds', None)
            scene_name = scene['scene_name']
            
            if dataset not in self.adapters:
                raise ValueError(f"Adapter for {dataset} not registered.")
                
            eeg, ya, yb, label = self.adapters[dataset].get_segment(subject, trial, start_sec, dur_sec)
            
            eeg_list.append(eeg)
            ya_list.append(ya)
            yb_list.append(yb)
            
            num_samples = eeg.shape[1]
            for _ in range(num_samples):
                window_metadata.append({
                    'scene_name': scene_name,
                    'scenario_name': scenario_name,
                    'dataset': dataset,
                    'subject': subject,
                    'trial': trial,
                    'ground_truth': label
                })
                
        # Concatenate along time axis
        full_eeg = np.concatenate(eeg_list, axis=1)
        full_ya = np.concatenate(ya_list, axis=0)
        full_yb = np.concatenate(yb_list, axis=0)
        
        # 2. Yield Windows (Sliding over the concatenated array)
        win_samples = int(self.window_sec * self.fs)
        hop_samples = int(self.hop_sec * self.fs)
        
        start = 0
        window_idx = 0
        
        while start + win_samples <= full_eeg.shape[1]:
            end = start + win_samples
            
            e_win = full_eeg[:, start:end]
            a_win = full_ya[start:end]
            b_win = full_yb[start:end]
            
            # Metadata for this window corresponds to the exact center of the window
            center_sample = start + (win_samples // 2)
            meta = window_metadata[center_sample]
            
            yield {
                'eeg_window': e_win,
                'audio_a_window': a_win,
                'audio_b_window': b_win,
                'ground_truth': meta['ground_truth'],
                'timestamp_sec': center_sample / self.fs,
                'scene_name': meta['scene_name'],
                'scenario_name': meta['scenario_name'],
                'dataset': meta['dataset'],
                'subject': meta['subject'],
                'trial': meta['trial'],
                'window_idx': window_idx
            }
            
            start += hop_samples
            window_idx += 1
