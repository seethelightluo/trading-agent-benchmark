import pandas as pd, os
# check volume availability in stock_data
for s in ['000300.SH','SPX','BTC','US10Y','XAU']:
    df = pd.read_csv(f'../persistent/stock_data/{s}.csv', parse_dates=['date'])
    has_vol = 'volume' in df.columns
    if has_vol:
        v = df['volume']
        nz = (v > 0).mean()
        print(s, 'volume present, non-zero frac:', round(nz,3), 'last vol:', v.iloc[-1])
    else:
        print(s, 'NO volume column')
# check OHLC columns
print(df.columns.tolist())
