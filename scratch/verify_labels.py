import sys
import json
import random
from pathlib import Path
from pprint import pprint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines.ridge_aad import subject_files, load_subject_examples

def get_speaker_gender(filename):
    filename = filename.lower()
    male_speakers = ["aske", "carsten", "jesper", "thomas", "oliver", "peter", "jens", "christian"]
    female_speakers = ["marianne", "pernille", "susanne", "trine", "anna", "maria", "mette", "karen", "stine", "line", "ida"]
    
    for m in male_speakers:
        if m in filename:
            return "Male"
    for f in female_speakers:
        if f in filename:
            return "Female"
            
    return "Unknown"

def main():
    mapping_path = Path("data/audio_mapping.json")
    with open(mapping_path, "r") as f:
        mapping = json.load(f)
        
    all_paths = subject_files()
    
    trials_to_print = []
    
    for p in all_paths:
        subject_id = p.stem.replace("_data_preproc", "")
        examples = load_subject_examples(p)
        
        # Pick 2 random trials from each subject (to get ~36 trials total)
        if len(examples) > 0:
            sample_trials = random.sample(examples, min(2, len(examples)))
            
            for ex in sample_trials:
                trial_key = f"trial_{ex.trial_index}"
                
                if subject_id in mapping and trial_key in mapping[subject_id]:
                    wav_a_file = mapping[subject_id][trial_key]["wavA"]["filename"]
                    wav_b_file = mapping[subject_id][trial_key]["wavB"]["filename"]
                    
                    wav_a_gender = get_speaker_gender(wav_a_file)
                    wav_b_gender = get_speaker_gender(wav_b_file)
                    
                    label = ex.label
                    
                    # Assume label 1 = Male attended, 2 = Female attended
                    expected_attended_gender = "Male" if label == 1 else "Female" if label == 2 else str(label)
                    
                    is_wavA_attended = (wav_a_gender == expected_attended_gender)
                    
                    trials_to_print.append({
                        "subject": subject_id,
                        "trial": ex.trial_index,
                        "label": label,
                        "expected_attended": expected_attended_gender,
                        "wavA_file": wav_a_file,
                        "wavA_gender": wav_a_gender,
                        "wavB_file": wav_b_file,
                        "wavB_gender": wav_b_gender,
                        "is_wavA_attended": is_wavA_attended
                    })
                    
    # Print the requested info
    print(f"{'Subject':<8} {'Trial':<8} {'Label':<6} {'Expected':<10} {'wavA File':<30} {'wavA_Gender':<12} {'wavB File':<30} {'wavB_Gender':<12} {'wavA==Attended?':<15}")
    print("-" * 140)
    for t in trials_to_print[:30]:
        print(f"{t['subject']:<8} {t['trial']:<8} {t['label']:<6} {t['expected_attended']:<10} {t['wavA_file']:<30} {t['wavA_gender']:<12} {t['wavB_file']:<30} {t['wavB_gender']:<12} {str(t['is_wavA_attended']):<15}")
        
    print("\n--- SUMMARY ---")
    total = len(trials_to_print)
    wavA_is_attended = sum(1 for t in trials_to_print if t['is_wavA_attended'])
    print(f"Total checked trials: {total}")
    print(f"wavA == attended speaker: {wavA_is_attended} / {total} ({wavA_is_attended/total*100:.2f}%)")

if __name__ == "__main__":
    main()
