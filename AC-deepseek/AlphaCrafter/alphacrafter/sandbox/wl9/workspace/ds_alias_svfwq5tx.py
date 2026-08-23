import pandas as pd
# evaluate recent regime across the tradable universe from persistent stock_data
files=['000300.SH','000688.SH','BTC','CN10Y','COPPER','ETH','HSI','N225','NDX','SOX','SPX','SX5E','WTI','XAU','US10Y']
for f in ['SPX','SX5E','WTI','XAU','US10Y']:
    try:
        df=pd.read_csv(f'../persistent/stock_data/{f}.csv')
        print(f, df['date'].iloc[-1], "rows", len(df))
    except Exception as e:
        print(f, "ERRNAME", e)