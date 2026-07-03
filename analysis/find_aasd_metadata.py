import os
import glob

def main():
    roots = [
        "/kaggle/input/datasets/lokeshgile/aasd-processed-eeg",
        "/kaggle/input/datasets/lokeshgile/aasd-audio-gammatones"
    ]
    
    for r in roots:
        print(f"\nSearching metadata in {r}...")
        for root, dirs, files in os.walk(r):
            for f in files:
                if f.endswith('.txt') or f.endswith('.json') or f.endswith('.csv') or f.endswith('.md') or f.endswith('.mat'):
                    if 'S18' in root or 'S1' in root:
                        continue # Skip individual subject mat files for brevity
                    print(os.path.join(root, f))
                    
if __name__ == "__main__":
    main()
