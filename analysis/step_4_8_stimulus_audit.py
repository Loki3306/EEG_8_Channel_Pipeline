import os
import scipy.io as sio
import pandas as pd
import numpy as np
import argparse

def find_mat_files(base_path):
    mat_files = []
    for root, dirs, files in os.walk(base_path):
        for f in files:
            if f.endswith('.mat'):
                mat_files.append(os.path.join(root, f))
    return sorted(mat_files)

def extract_expinfo(mat_files):
    all_data = []
    
    for mat_path in mat_files:
        try:
            basename = os.path.basename(mat_path)
            subject_id = basename.split('.')[0]
            
            mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=True)
            
            if 'expinfo' in mat:
                expinfo = mat['expinfo']
                
                if hasattr(expinfo, 'dtype') and expinfo.dtype.names:
                    names = expinfo.dtype.names
                    
                    if 'wavfile_male' in names and 'wavfile_female' in names:
                        # Depending on MATLAB version, expinfo might be 0D array or 1D array of structs
                        if expinfo.ndim == 0:
                             expinfo = np.atleast_1d(expinfo)
                        
                        n_trials = len(expinfo)
                        
                        for i in range(n_trials):
                            try:
                                trial_record = expinfo[i]
                                
                                # Use item() if scalar numpy types
                                male_wav = str(trial_record['wavfile_male']).strip()
                                female_wav = str(trial_record['wavfile_female']).strip()
                                
                                attend_mf = str(trial_record['attend_mf']).strip() if 'attend_mf' in names else 'unknown'
                                
                                all_data.append({
                                    'subject_id': subject_id,
                                    'trial_id': i + 1,
                                    'male_wav': male_wav,
                                    'female_wav': female_wav,
                                    'attend_mf': attend_mf
                                })
                            except Exception as e:
                                pass
        except Exception as e:
            pass
            
    return pd.DataFrame(all_data)

def generate_report(df):
    if df.empty:
        print("No trial mapping data found!")
        return
        
    print("\n--- STIMULUS REUSE METADATA AUDIT ---")
    print(f"Total mapped trials found: {len(df)}")
    print(f"Unique Subjects: {df['subject_id'].nunique()}")
    
    unique_male = df['male_wav'].nunique()
    unique_female = df['female_wav'].nunique()
    
    print(f"\nUnique male audio files: {unique_male}")
    print(f"Unique female audio files: {unique_female}")
    
    # How many subjects hear each chunk?
    male_reuse = df.groupby("male_wav")['subject_id'].nunique().sort_values(ascending=False)
    female_reuse = df.groupby("female_wav")['subject_id'].nunique().sort_values(ascending=False)
    
    print("\n--- AUDIO REUSE (MALE WAVS) ---")
    print(male_reuse.head(10))
    print(f"... and {max(0, len(male_reuse) - 10)} more.")
    
    print("\n--- AUDIO REUSE (FEMALE WAVS) ---")
    print(female_reuse.head(10))
    print(f"... and {max(0, len(female_reuse) - 10)} more.")
    
    # Are stories identical across subjects?
    subject_wavs = df.groupby('subject_id').apply(lambda g: set(g['male_wav']) | set(g['female_wav'])).reset_index()
    subject_wavs.columns = ['subject_id', 'wav_set']
    
    subject_wavs['wav_tuple'] = subject_wavs['wav_set'].apply(lambda x: tuple(sorted(list(x))))
    unique_sets = subject_wavs.groupby('wav_tuple').size()
    
    print(f"\nNumber of distinct full-experiment audio sets: {len(unique_sets)}")
    
    print("\n--- FINAL AUDIT ANSWERS ---")
    print("1. Does every subject hear the same stories?")
    if len(unique_sets) == 1:
        print("   YES. Every subject heard the exact same set of audio files.")
    else:
        print(f"   NO. There are {len(unique_sets)} different experiment variants.")
        
    print("2. Is Leave-One-Story-Out feasible?")
    total_audio = unique_male + unique_female
    if total_audio < 20:
        print(f"   LIKELY NO. There are only {total_audio} total audio chunks. Leaving one out starves the model.")
    else:
        print(f"   POTENTIALLY. There are {total_audio} total audio chunks.")
        
    print("3. Could stimulus reuse plausibly explain the confidence results?")
    if df['subject_id'].nunique() > 0 and male_reuse.mean() > df['subject_id'].nunique() * 0.5:
         print("   YES. Average reuse is > 50% of subjects. Acoustic fingerprinting is mathematically possible.")
    else:
         print("   WEAKLY. Most audio is unique or sparsely reused.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="/kaggle/input/datasets/lokeshgile/eeg-audio")
    args = parser.parse_args()
    
    mat_files = find_mat_files(args.dir)
    print(f"Found {len(mat_files)} .mat files. Extracting expinfo...")
    
    df = extract_expinfo(mat_files)
    if not df.empty:
        df.to_csv("audio_mapping_audit.csv", index=False)
        print("Saved raw mapping to audio_mapping_audit.csv")
        generate_report(df)
    else:
        print("Failed to extract expinfo matching the expected structure.")
        
        # Fallback diagnostic
        print("\nFallback Diagnostic: Printing keys for first MAT file...")
        if len(mat_files) > 0:
            mat = sio.loadmat(mat_files[0], squeeze_me=True, struct_as_record=True)
            print(f"Keys: {mat.keys()}")
            if 'expinfo' in mat:
                print(f"expinfo type: {type(mat['expinfo'])}")
                if hasattr(mat['expinfo'], 'dtype'):
                    print(f"expinfo dtype names: {mat['expinfo'].dtype.names}")

if __name__ == "__main__":
    main()
