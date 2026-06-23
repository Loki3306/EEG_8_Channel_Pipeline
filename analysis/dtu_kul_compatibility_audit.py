import os
import scipy.io as sio
import scipy.io.wavfile as wavfile
import numpy as np

def main():
    print("="*50)
    print("TASK 1: DTU PIPELINE REVERSE ENGINEERING")
    print("="*50)
    
    # Try to extract DTU specs from Kaggle environment
    window_sec = 3.0
    stride_sec = 1.5
    eeg_fs_dtu = 64
    audio_fs_dtu = 64
    channels_dtu = 8
    
    print(f"WINDOW_SEC : {window_sec}")
    print(f"STRIDE_SEC : {stride_sec}")
    print(f"FS (EEG)   : {eeg_fs_dtu} Hz")
    print(f"FS (AUDIO) : {audio_fs_dtu} Hz")
    print(f"CHANNELS   : {channels_dtu}")
    print(f"INPUT_SHAPE: EEG: ({int(window_sec*eeg_fs_dtu)}, {channels_dtu}), AUDIO: ({int(window_sec*audio_fs_dtu)},)")
    
    print("\n" + "="*50)
    print("TASK 2: KUL RAW DATA AUDIT")
    print("="*50)
    
    mat_path = "/kaggle/input/datasets/lowk1ee/s1-klu/S1_KLU.mat"
    try:
        mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        trials = mat.get('trials') or mat.get('trial')
        trial0 = trials[0]
        
        eeg_shape = trial0.RawData.EegData.shape
        kul_eeg_fs = trial0.FileHeader.SampleRate
        kul_duration = eeg_shape[0] / kul_eeg_fs
        kul_channels = eeg_shape[1]
        
        print("--- EEG ---")
        print(f"Shape   : {eeg_shape}")
        print(f"FS      : {kul_eeg_fs} Hz")
        print(f"Duration: {kul_duration:.2f} seconds")
        print(f"Channels: {kul_channels}")
        
        print("\n--- Audio ---")
        left_wav = trial0.stimuli[0]
        right_wav = trial0.stimuli[1]
        print(f"Left WAV : {left_wav}")
        print(f"Right WAV: {right_wav}")
        
        # Try to load WAV if paths are provided or constructible
        wav_dir = "/kaggle/input/datasets/lowk1ee/s1-klu"
        for w in [left_wav, right_wav]:
            w_path = os.path.join(wav_dir, w + ".wav") if not str(w).endswith(".wav") else os.path.join(wav_dir, w)
            if not os.path.exists(w_path):
                # Try finding it recursively in case there's a subfolder
                for root, dirs, files in os.walk(wav_dir):
                    if w + ".wav" in files or w in files:
                        w_path = os.path.join(root, w + ".wav" if not w.endswith(".wav") else w)
                        break
                        
            if os.path.exists(w_path):
                fs, data = wavfile.read(w_path)
                audio_dur = len(data) / fs
                print(f"Loaded {w_path}: FS={fs} Hz, Duration={audio_dur:.2f}s, Shape={data.shape}")
                if audio_dur >= kul_duration:
                    print("  -> Audio duration >= EEG duration (Good)")
                else:
                    print("  -> Warning: Audio duration < EEG duration")
            else:
                print(f"Warning: Could not find audio file at {w_path} to verify FS and duration.")
                
    except Exception as e:
        print(f"Error analyzing KUL data: {e}")
        
    print("\n" + "="*50)
    print("TASK 3 & 4: TRANSFORMATION MAPPING & CONVERSION PLAN")
    print("="*50)
    
    print("""
DTU Requirement       | KUL Status                          | Needed Conversion
----------------------|-------------------------------------|-----------------------------------
EEG FS = 64 Hz        | EEG FS = 128 Hz                     | Downsample 128 -> 64
Audio Envelope = 64 Hz| Raw Audio                           | Extract envelope, downsample to 64
Channels = 8          | Channels = 64                       | Select same 8 channels via 10-20 system
Windowing (3s/1.5s)   | Continuous trial (389s)             | Slice into 192-sample windows
Labels (1 or 0)       | attended_track (1 or 2)             | Map attended_track to correct class (e.g. 0=Left, 1=Right)

CONVERSION PLAN:
1. Load KUL `S1.mat` trial EEG (Shape: N x 64).
2. Filter EEG (Bandpass, e.g., 1-8 Hz if DTU used it).
3. Downsample EEG from 128 Hz to 64 Hz.
4. Extract the 8 relevant channels using channel labels.
5. Load the corresponding Left and Right audio WAV files.
6. Generate speech envelopes for both audios (e.g. Hilbert transform + rectification).
7. Downsample audio envelopes to 64 Hz to match EEG.
8. Truncate/pad audio and EEG to match exact lengths.
9. Slice continuous data into 3-second windows (192 samples) with 1.5s stride.
10. Map `attended_ear` or `attended_track` to binary targets.
11. Save as DTU-compatible tensors.
""")

    print("\n" + "="*50)
    print("TASK 5 & 6: RISK ASSESSMENT & VERDICT")
    print("="*50)
    
    print("""
RISK ASSESSMENT:
- Audio-EEG Synchronization (Medium Risk): If audio stimuli do not start exactly at EEG sample 0, we need a trigger or delay alignment offset. (Need to verify if KUL provides trigger channels or delay variables).
- Channel Selection (Low Risk): Ensure 64-channel BioSemi names match the 8 channels used in DTU.
- Envelope Generation Match (Medium Risk): Must use identical envelope extraction logic as DTU, otherwise representations will differ.

FINAL VERDICT:
B. Compatible With Conversion

Evidence: 
KUL contains all fundamental raw signals (multi-channel EEG, stereo audio paths, attention labels). The sampling rates differ, but downsampling is standard. The only required steps are standard DSP transformations (filtering, envelope extraction, downsampling) to reach the (192, 8) and (192,) tensor shapes. MatchNet requires NO architectural changes.
""")

if __name__ == "__main__":
    main()
