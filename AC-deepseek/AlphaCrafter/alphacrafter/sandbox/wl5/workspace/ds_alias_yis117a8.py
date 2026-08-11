import pandas as pd, os
WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
for s in WATCH:
    df = pd.read_csv(f'../persistent/stock_data/{s}.csv', parse_dates=['date'])
    has_vol = 'volume' in df.columns
    nz = (df['volume'] > 0).mean() if has_vol else 0
    pe = df['PE'].notna().mean() if 'PE' in df.columns else 0
    print(f"{s:10s} vol_nz={nz:.2f} PE_cov={pe:.2f} rows={len(df)}")
