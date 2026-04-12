import pandas as pd
df = pd.read_csv('data/historical.csv')
stats = df.groupby('segment')['balance'].agg(['min', 'mean', 'max']).round(0)
print(stats.to_string())
print("\nAny negative balances:", (df['balance'] < 0).sum())