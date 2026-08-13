import pandas as pd
for sym in ['SPX','BTC','COPPER','000300.SH','US10Y','CN10Y','XAU','ETH']:
    df = pd.read_csv(f'../persistent/stock_data/{sym}.csv')
    print(sym, df.shape, str(df['date'].iloc[0])[:10], '->', str(df['date'].iloc[-1])[:10])
print()
df = pd.read_csv('../persistent/index_data/VIX.csv')
print('VIX', df.shape, str(df['date'].iloc[0])[:10], '->', str(df['date'].iloc[-1])[:10])
print(df.columns.tolist())