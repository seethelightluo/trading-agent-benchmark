import pandas as pd
for sym in ['SPX','NDX','SOX','N225','SX5E','000300.SH','000688.SH','HSI','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']:
    df = pd.read_csv(f'../persistent/stock_data/{sym}.csv')
    print(sym, df.shape, df['date'].iloc[0], '->', df['date'].iloc[-1])
