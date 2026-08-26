import pandas as pd, os
# date range of data
for s in ['SPX','BTC','XAU','US10Y','CN10Y','WTI','000300.SH','ETH','NDX','HSI','COPPER']:
    df = pd.read_csv(f'../persistent/stock_data/{s}.csv')
    print(s, df.shape, df.columns.tolist()[:8], str(df.iloc[0,0]), '->', str(df.iloc[-1,0]))
print('---INDEX---')
for s in ['DXY','VIX','USDJPY','USDCNY','EURUSD']:
    df = pd.read_csv(f'../persistent/index_data/{s}.csv')
    print(s, df.shape, str(df.iloc[0,0]), '->', str(df.iloc[-1,0]))