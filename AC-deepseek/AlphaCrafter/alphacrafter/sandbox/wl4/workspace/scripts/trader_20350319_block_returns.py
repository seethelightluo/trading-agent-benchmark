import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

assets = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def get(a, n=40):
    df = get_stock_daily_data(a, days=n)
    if df is None or len(df)==0:
        df = get_index_daily_data(a, days=n)
    return df

rows = {}
for a in assets:
    df = get(a)
    if df is None or len(df)==0:
        print(a, 'NO DATA')
        continue
    df = df.sort_values('date').reset_index(drop=True)
    dates = [d.strftime('%Y-%m-%d') for d in df['date']]
    closes = df['close'].astype(float)
    def idx(d):
        for i in range(len(dates)-1, -1, -1):
            if dates[i] <= d:
                return i
        return None
    i0 = idx('2035-03-16')
    i1 = idx('2035-04-02')
    if i0 is None or i1 is None or i1 <= i0:
        print(a, 'range issue', dates[-3:] if dates else None)
        continue
    r = closes.iloc[i1]/closes.iloc[i0] - 1.0
    rows[a] = (dates[i0], dates[i1], r*100, closes.iloc[i0], closes.iloc[i1])

print(f"{'asset':10s} {'from':10s} {'to':10s} {'ret%':>8s} {'c0':>10s} {'c1':>10s}")
for a in assets:
    if a in rows:
        d0, d1, r, c0, c1 = rows[a]
        print(f"{a:10s} {d0:10s} {d1:10s} {r:8.2f} {c0:10.4f} {c1:10.4f}")
