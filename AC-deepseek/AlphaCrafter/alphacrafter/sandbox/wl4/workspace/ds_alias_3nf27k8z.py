import pandas as pd, numpy as np
syms = ['SPX','NDX','SOX','N225','SX5E','000300.SH','000688.SH','HSI','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
prices = {}
for s in syms:
    df = pd.read_csv(f'../persistent/stock_data/{s}.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= '2032-12-12'].set_index('date').sort_index()
    prices[s] = df['close']
px = pd.DataFrame(prices).dropna(how='all')
print('rows:', px.shape, px.index.min(), '->', px.index.max())
ret = px.pct_change()
print('\n=== Last close ===')
print(px.iloc[-1].round(2))
print('\n=== Returns since 2032-09-20 (last cycle end) ===')
sub = px[px.index >= '2032-09-20']
print((sub.iloc[-1]/sub.iloc[0]-1).round(4))
print('\n=== 60d return (to 12-12) ===')
sub60 = px[px.index >= px.index[-1] - pd.Timedelta(days=90)]
print((sub60.iloc[-1]/sub60.iloc[0]-1).round(4))
