import pandas as pd, os
WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
for s in WATCH:
    df = pd.read_csv(f'../persistent/stock_data/{s}.csv', parse_dates=['date'])
    ohlc = all(c in df.columns for c in ['open','high','low','close'])
    nz_h = (df['high'] > 0).mean()
    nz_o = (df['open'] > 0).mean()
    print(f"{s:10s} ohlc={ohlc} open_nz={nz_o:.2f} high_nz={nz_h:.2f}")
