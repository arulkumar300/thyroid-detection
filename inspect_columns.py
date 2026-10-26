import pandas as pd
df = pd.read_csv(r'C:\Users\elango\Downloads\thyroidDF (4).csv')
print("Shape:", df.shape)
for col in df.columns:
    if df[col].dtype == 'object':
        print(f"Column '{col}': unique values = {df[col].unique()[:20]}")
