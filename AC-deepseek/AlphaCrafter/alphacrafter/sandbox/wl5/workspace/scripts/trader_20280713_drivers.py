"""Compute block return drivers 2028-06-28 -> 2028-07-12 (10 trading days)."""
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
import pandas as pd

WATCH = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
         'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']

def closes(sym, days=25):
    df = get_stock_daily_data(sym, days=days)
    if df is None or len(df) < 15:
        df = get_index_daily_data(sym, days=days)
    if df is None or len(df) < 15:
        return None
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    return df.set_index('date')['close'].astype(float)

print(f"{'sym':8s} {'px_06-28':>12s} {'px_07-12':>12s} {'block_ret':>10s}")
rows = []
for s in WATCH:
    c = closes(s)
    if c is None:
        continue
    p0 = float(c.iloc[-11])  # 10 trading days back
    p1 = float(c.iloc[-1])
    r = p1 / p0 - 1.0
    rows.append((s, r))
    print(f"{s:8s} {p0:12.4f} {p1:12.4f} {r*100:9.2f}%")

rows.sort(key=lambda x: x[1], reverse=True)
print("\nWinners:", ", ".join(f"{s} {r*100:+.2f}%" for s, r in rows[:5]))
print("Laggards:", ", ".join(f"{s} {r*100:+.2f}%" for s, r in rows[-5:]))
