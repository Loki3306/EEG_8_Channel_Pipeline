import os
import scipy.io

def get_labels_from_mat(mat_path):
    try:
        mat = scipy.io.loadmat(mat_path, simplify_cells=True)
        if 'EEG' in mat and 'chanlocs' in mat['EEG']:
            chanlocs = mat['EEG']['chanlocs']
            if isinstance(chanlocs, list):
                return [c['labels'] for c in chanlocs]
            elif isinstance(chanlocs, dict) and 'labels' in chanlocs:
                # If there's only one channel, it might be a dict, but usually it's a list
                pass
        
        # Alternative EEGLAB struct
        if 'chanlocs' in mat:
            pass
            
    except Exception as e:
        print(f"Error reading {mat_path}: {e}")
    return None

def main():
    print("--- 1. Searching for Dataset MAT Files ---")
    
    # Common Kaggle paths for KUL and DTU
    kul_dtu_paths = [
        '/kaggle/input/dtu-dataset/S1.mat',
        '/kaggle/input/dtu-processed/S1.mat',
        '/kaggle/input/kul-dataset/S1.mat',
        '/kaggle/input/kul-processed/S1.mat',
    ]
    
    # KUL/DTU used indices [13, 46, 43, 23, 50, 0, 52, 14]
    target_indices = [13, 46, 43, 23, 50, 0, 52, 14]
    kul_labels = None
    
    for p in kul_dtu_paths:
        if os.path.exists(p):
            print(f"Found KUL/DTU file: {p}")
            labels = get_labels_from_mat(p)
            if labels and len(labels) >= max(target_indices):
                kul_labels = [labels[i] for i in target_indices]
                print(f"\n=> KUL/DTU 8 Channels: {kul_labels}")
                break
                
    if not kul_labels:
        print("\n[WARNING] Could not find KUL/DTU .mat files on Kaggle to extract labels.")
        print("Assuming standard BioSemi 64 mapping for DTU: ['Cz', 'POz', 'P3', 'P4', 'O1', 'Fp1', 'O2', 'Pz']")
        kul_labels = ['Cz', 'POz', 'P3', 'P4', 'O1', 'Fp1', 'O2', 'Pz'] # Educated guess for those indices
        
    print("\n--- 2. Searching for AASD MAT Files ---")
    aasd_path = '/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG/S18/S18.mat'
    if not os.path.exists(aasd_path):
        print(f"ERROR: Could not find {aasd_path}")
        return
        
    aasd_labels = get_labels_from_mat(aasd_path)
    if not aasd_labels:
        print(f"ERROR: Could not extract chanlocs from {aasd_path}")
        return
        
    print(f"Found {len(aasd_labels)} channels in AASD.")
    print(f"First 10 AASD labels: {aasd_labels[:10]}")
    
    print("\n--- 3. Mapping KUL -> AASD ---")
    aasd_indices = []
    missing = []
    
    for lbl in kul_labels:
        # Match case-insensitive
        match = next((i for i, a_lbl in enumerate(aasd_labels) if str(a_lbl).lower() == str(lbl).lower()), None)
        if match is not None:
            aasd_indices.append(match)
            print(f"Matched '{lbl}' -> AASD Index {match}")
        else:
            missing.append(lbl)
            print(f"FAILED to match '{lbl}'")
            
    if missing:
        print(f"\n[WARNING] Could not find exact matches for {missing}.")
        print("This means the channel naming conventions differ (e.g. 'A1' vs 'Fp1').")
        print("Please check the AASD label list above and manually map them.")
    else:
        print("\nSUCCESS! Paste this into phase39_transfer_ridge_hybrid.py:")
        print(f"PHYSICAL_8_CHANNELS = {aasd_indices}")

if __name__ == "__main__":
    main()
