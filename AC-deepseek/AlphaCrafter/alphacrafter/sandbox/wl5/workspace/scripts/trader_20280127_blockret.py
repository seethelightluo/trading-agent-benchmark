from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
import pandas as pd

assets = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def closes(a):
    df = get_stock_daily_data(a, days=40)
    if df is None or len(df) < 30:
        df = get_index_daily_data(a, days=40)
    df = df[['date','close']].copy()
    df['date'] = pd.to_datetime(df['date'])
    return df.set_index('date')['close'].astype(float)

rows = []
for a in assets:
    c = closes(a)
    if c is None or len(c) < 25:
        print(a, "insufficient data", 0 if c is None else len(c))
        continue
    p0 = float(c.iloc[-11])   # close on/before 2028-01-26 (start of block)
    p1 = float(c.iloc[-1])    # latest close (2028-02-09/10)
    r = p1/p0 - 1.0
    rows.append((a, p0, p1, r))
    print(f"{a:10s} p_start={p0:>12.4f} p_end={p1:>12.4f} block_ret={r*100:>8.2f}%")
