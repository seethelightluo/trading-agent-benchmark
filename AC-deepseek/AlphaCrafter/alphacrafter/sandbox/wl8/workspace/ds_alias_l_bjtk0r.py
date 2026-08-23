import pandas as pd, os
for f in ['SPX.csv','BTC.csv','US10Y.csv','CN10Y.csv','000300.SH.csv']:
    df = pd.read_csv('../persistent/stock_data/'+f)
    print(f, len(df), 'cols:', list(df.columns)[:8], 'end:', str(df.iloc[-1,0])[:10])