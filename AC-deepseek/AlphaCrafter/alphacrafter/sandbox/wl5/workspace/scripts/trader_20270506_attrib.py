"""Attribute block P&L 2027-04-22 -> 2027-05-06 and check rebalance status."""
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
import pandas as pd

ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU',
          'COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def closes(a, days=12):
    df = get_stock_daily_data(a, days=days)
    if df is None or len(df) < 5:
        df = get_index_daily_data(a, days=days)
    if df is None or len(df) < 5:
        return None
    s = df[['date','close']].copy()
    s['date'] = pd.to_datetime(s['date'])
    return s.set_index('date')['close']

print("Per-asset block return (2027-04-21 close -> 2027-05-06 close):")
rows = []
for a in ASSETS:
    c = closes(a)
    if c is None:
        print(f"{a:10s} NO DATA")
        continue
    p0 = float(c.iloc[-2])   # 2027-04-21 (block start eve)
    p1 = float(c.iloc[-1])   # 2027-05-06
    ret = p1 / p0 - 1.0
    rows.append((a, ret, p0, p1))
    print(f"{a:10s} ret={ret:+.4f}  ({p0:.2f} -> {p1:.2f})")

# approx weights at block start from last summary (drifted post-04-08 rebalance)
w0 = {'000300.SH':0.0647,'SPX':0.056,'HSI':0.058,'N225':0.057,'SX5E':0.055,
      '000688.SH':0.0647,'SOX':0.049,'NDX':0.067,'XAU':0.058,'COPPER':0.053,
      'WTI':0.079,'BTC':0.104,'ETH':0.095,'US10Y':0.059,'CN10Y':0.0647}
tot = 0.0
for a, ret, _, _ in rows:
    tot += w0.get(a, 0.06) * ret
print(f"\nApprox block contribution (using prior drifted weights): {tot:+.4f}")
