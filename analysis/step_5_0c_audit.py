import pandas as pd
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="subject_distance_predictions.csv")
    args = parser.parse_args()
    
    try:
        df = pd.read_csv(args.csv)
    except FileNotFoundError:
        print(f"File {args.csv} not found.")
        return
        
    print("\n--- TRIAL LENGTH AUDIT ---")
    desc = df.groupby(["subject_id", "trial_id"]).size().describe()
    print(desc)
    
    print("\nTotal Subjects:", df['subject_id'].nunique())
    print("Total Trials:", len(df.groupby(["subject_id", "trial_id"])))
    print("Total Windows:", len(df))

if __name__ == "__main__":
    main()
