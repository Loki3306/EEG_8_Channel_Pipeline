"""
raw_eeg_utils.py
Helper functions to load and parse FieldTrip-style raw EEG .mat files
in Python, handling both scipy.io (MATLAB v7.2) and h5py (MATLAB v7.3).
"""

import h5py
import scipy.io as sio
import numpy as np

def load_raw_eeg(mat_path):
    """
    Attempts to load the raw EEG data assuming a FieldTrip structure named 'data'.
    Returns: dict with keys:
        - 'fsample': int/float
        - 'labels': list of str (channel names)
        - 'trials': list of np.ndarray, each (channels, time)
    """
    try:
        mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        return _parse_scipy_ft(mat['data'])
    except NotImplementedError:
        # HDF5 format
        with h5py.File(mat_path, 'r') as f:
            return _parse_h5py_ft(f['data'], f)
            
def _parse_scipy_ft(data):
    # data is a scipy.io.matlab.mio5_params.mat_struct
    # Extract fsample
    fsample = float(data.fsample.eeg if hasattr(data.fsample, 'eeg') else data.fsample)
    
    # Extract labels
    labels = []
    if hasattr(data, 'label'):
        labels = [str(l) for l in data.label]
        
    # Extract trials
    trials = []
    if hasattr(data, 'trial'):
        # Usually a cell array of matrices
        if isinstance(data.trial, np.ndarray):
            if data.trial.dtype == object:
                trials = [tr for tr in data.trial]
            else:
                trials = [data.trial]
                
    return {"fsample": fsample, "labels": labels, "trials": trials}

def _parse_h5py_ft(data_group, f):
    # HDF5 parsing for FieldTrip
    fsample_node = data_group.get('fsample')
    if 'eeg' in fsample_node:
        fsample = float(fsample_node['eeg'][0,0])
    else:
        fsample = float(fsample_node[0,0])
        
    # Extract labels
    labels = []
    if 'label' in data_group:
        refs = data_group['label'][:,0]
        for ref in refs:
            # dereference
            obj = f[ref]
            # convert uint16 array to string
            labels.append(''.join(chr(c[0]) for c in obj[:]))
            
    # Extract trials
    trials = []
    if 'trial' in data_group:
        refs = data_group['trial'][:,0]
        for ref in refs:
            trials.append(f[ref][...]) # (channels, time)
            
    return {"fsample": fsample, "labels": labels, "trials": trials}
