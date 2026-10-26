import pandas as pd
import numpy as np

def inspect_file(path):
    print(f"=== Inspecting {path} ===")
    try:
        df = pd.read_csv(path)
        print(f"Shape: {df.shape}")
        print("Columns:", list(df.columns))
        print("Target counts:")
        # Look for target-like columns
        target_cols = [c for c in df.columns if c.lower() in ['target', 'class', 'label']]
        for c in target_cols:
            print(f"  Column '{c}':")
            print(df[c].value_counts().head(10))
        # Missing values summary
        print("Null counts for continuous-like columns (top 5):")
        null_sums = df.isnull().sum()
        print(null_sums[null_sums > 0].sort_values(ascending=False).head(5))
    except Exception as e:
        print(f"Error reading {path}: {e}")
    print()

inspect_file(r'C:\Users\elango\Downloads\thyroidDF.csv')
inspect_file(r'C:\Users\elango\Downloads\thyroidDF (4).csv')
