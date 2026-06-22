import argparse
import pandas as pd
import numpy as np

def sanity_check(df):
    print("\n--- SANITY CHECK ---")
    print("\n1. DataFrame Head:")
    print(df.head())
    
    print(f"\n2. Missing Values (NaNs): {df.isna().sum().sum()}")
    
    print(f"\n3. Overall Accuracy: {df['correct'].mean():.4f} ({df['correct'].mean()*100:.2f}%)")
    
    print("\n4. Similarity A Description:")
    print(df['sim_A'].describe())
    
    print("\n5. Similarity B Description:")
    print(df['sim_B'].describe())

def step_1_2_margin_analysis(df):
    print("\n===========================================")
    print("STEP 1.2: MARGIN CONFIDENCE ANALYSIS")
    print("===========================================\n")
    
    # Generate margin
    df['margin'] = np.abs(df['sim_A'] - df['sim_B'])
    
    # --- Analysis 1 ---
    print("--- Analysis 1: Mean Margin (Correct vs Incorrect) ---")
    correct_margin = df[df['correct'] == 1]['margin'].mean()
    incorrect_margin = df[df['correct'] == 0]['margin'].mean()
    
    print(f"Correct Predictions Mean Margin   : {correct_margin:.4f}")
    print(f"Incorrect Predictions Mean Margin : {incorrect_margin:.4f}")
    
    if correct_margin > incorrect_margin:
        print("-> YES: Correct predictions are made with larger similarity margins.\n")
    else:
        print("-> NO: Correct predictions do NOT have larger margins.\n")
        
    # --- Analysis 2 ---
    print("--- Analysis 2: Accuracy per Margin Bin ---")
    
    # We will create fixed bins, e.g., 0.0-0.05, 0.05-0.10, etc.
    # To cover typical pearson correlation margins, let's use bins of width 0.05
    max_margin = df['margin'].max()
    bins = np.arange(0.0, max_margin + 0.05, 0.05)
    
    df['margin_bin'] = pd.cut(df['margin'], bins=bins, right=False)
    
    bin_stats = df.groupby('margin_bin', observed=True).agg(
        count=('correct', 'size'),
        accuracy=('correct', 'mean')
    ).reset_index()
    
    print(f"{'Bin (Margin)':<20} | {'Count':<8} | {'Accuracy':<10}")
    print("-" * 45)
    
    for _, row in bin_stats.iterrows():
        bin_str = str(row['margin_bin'])
        count = row['count']
        acc = row['accuracy'] * 100 if count > 0 else 0.0
        print(f"{bin_str:<20} | {count:<8} | {acc:.2f}%")
        
    print("\nDesired Pattern: Higher Margin -> Higher Accuracy")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="matchnet_predictions.csv")
    args = parser.parse_args()
    
    try:
        df = pd.read_csv(args.csv)
    except FileNotFoundError:
        print(f"Error: Could not find {args.csv}")
        return
        
    sanity_check(df)
    
    # If sanity check fails or has NaNs, we should theoretically stop, 
    # but we will print Step 1.2 regardless so the user can see everything in one go.
    step_1_2_margin_analysis(df)
    
    # Optionally save the df with the new margin column
    df.to_csv("matchnet_predictions_with_margin.csv", index=False)

if __name__ == "__main__":
    main()
