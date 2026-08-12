import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

def get(sym, n=220):
    try:
        df = get_stock_daily_data(sym, days=n)
    except Exception:
        df = None
    if df is None or len(df) < 60:
        try:
            df = get_index_daily_data(sym, days=n)
        except Exception:
            df = None
    return df

# Key assets to analyze block-by-block (10-trading-day non-overlapping blocks)
assets = ["N225", "XAU", "CN10Y", "US10Y", "WTI", "000688.SH", "SX5E", "SPX", "SOX", "COPPER", "BTC"]
closes = {}
for a in assets:
    df = get(a)
    if df is None:
        print(a, "NO DATA")
        continue
    df = df.sort_values("date").reset_index(drop=True)
    closes[a] = df["close"].astype(float)

# build common date index from SPX
spx = get("SPX").sort_values("date").reset_index(drop=True)
idx = spx["date"]
aligned = {}
for a, c in closes.items():
    s = pd.Series(c.values, index=closes[a].index)
    aligned[a] = pd.Series(s.values, index=idx.iloc[:len(s)].values) if len(s) <= len(idx) else None
    # simpler: reindex by position on the SPX calendar
    cc = pd.Series(c.values)
    aligned[a] = cc

n = len(spx)
print("SPX rows:", n, "first", spx["date"].iloc[0].date(), "last", spx["date"].iloc[-1].date())

# Block returns: last 6 blocks of 10 trading days (block k: close[k*10] -> close[(k+1)*10])
print("\nPer-block 10d returns (t-60d close -> latest):")
blocks = []
for k in range(6, 0, -1):
    i0 = n - (k+1)*10
    i1 = n - k*10
    if i0 < 0:
        continue
    d0 = spx["date"].iloc[i0].date()
    d1 = spx["date"].iloc[i1].date()
    blocks.append((d0, d1, i0, i1))

# also latest partial block (decision day 04-24 => index n-10 to n-1)
last0 = n - 10
last1 = n - 1
print(f"Block [04-24..05-08 proxy]: {spx['date'].iloc[last0].date()} -> {spx['date'].iloc[last1].date()} (closes {last0}->{last1})")

for (d0, d1, i0, i1) in blocks[-4:] + [(spx['date'].iloc[last0].date(), spx['date'].iloc[last1].date(), last0, last1)]:
    print(f"\n  {d0} -> {d1}:")
    for a in assets:
        c = aligned.get(a)
        if c is None or i1 >= len(c) or i0 >= len(c):
            print(f"    {a:10s} NA")
            continue
        p0 = float(c.iloc[i0]); p1 = float(c.iloc[i1])
        if p0 <= 0:
            continue
        print(f"    {a:10s} {p0:12.4f} -> {p1:12.4f}  {((p1/p0)-1)*100:+7.2f}%")
