import os
import sys
import numpy as np

def get_story_id(meta):
    """
    Extracts a unique Story ID from a trial's metadata by combining its 
    left and right stimuli filenames.
    
    The tuple is sorted to ensure that left/right permutations 
    (though unlikely) yield the same story identity if the underlying 
    audio mix is the same.
    """
    left = meta.get("stimuli_left", "unknown_left")
    right = meta.get("stimuli_right", "unknown_right")
    
    if left == "unknown_left" or right == "unknown_right":
        raise ValueError("stimuli_left or stimuli_right missing in meta. Run build_kul_cache.py to update cache.")
        
    # Sort them to guarantee order invariance
    pair = sorted([left, right])
    return f"{pair[0]}_AND_{pair[1]}"

def iter_leave_one_story_out(all_subject_data):
    """
    Given a dictionary of {subject_id: [trial_dicts]}, yields:
    (held_out_story_id, train_examples, test_examples)
    
    Each fold holds out ALL trials across ALL subjects that share the same Story ID,
    guaranteeing strict isolation of audio content between train and test.
    """
    # 1. Flatten all trials and collect unique story IDs
    all_trials = []
    unique_stories = set()
    
    for sub_id, trials in all_subject_data.items():
        for t in trials:
            story_id = get_story_id(t["meta"])
            unique_stories.add(story_id)
            all_trials.append({
                "sub_id": sub_id,
                "trial": t,
                "story_id": story_id
            })
            
    unique_stories = sorted(list(unique_stories))
    
    print(f"\n[LOStO] Found {len(unique_stories)} unique Story IDs across {len(all_trials)} total trials.")
    
    for held_out_story in unique_stories:
        train_examples = []
        test_examples = []
        
        train_story_ids = set()
        test_story_ids = set()
        
        for record in all_trials:
            if record["story_id"] == held_out_story:
                test_examples.append(record)
                test_story_ids.add(record["story_id"])
            else:
                train_examples.append(record)
                train_story_ids.add(record["story_id"])
                
        # Strict Leakage Verification
        intersection = train_story_ids.intersection(test_story_ids)
        assert len(intersection) == 0, f"LEAKAGE DETECTED! Stories {intersection} appear in both train and test."
        
        print(f"\n[LOStO] ----------------------------------------------------")
        print(f"[LOStO] Fold: {held_out_story}")
        print(f"[LOStO]   Train Stories: {len(train_story_ids)} | Test Stories: {len(test_story_ids)}")
        print(f"[LOStO]   Train Trials:  {len(train_examples)} | Test Trials:  {len(test_examples)}")
        print(f"[LOStO] ----------------------------------------------------")
        
        yield held_out_story, train_examples, test_examples
