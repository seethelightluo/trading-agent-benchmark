import pandas as pd

for f in ['../persistent/index_data/VIX.csv', '../persistent/stock_data/SPX.csv', '../persistent/stock_data/US10Y.csv']:
    df = pd.read_csv(f)
    print(f)
    print(df.columns.tolist())
    print(df.tail(3))
    print('len', len(df))
    print()