"""miner_1 2032-10-14: probe data availability and recent market state."""
import sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
from factor_common import WATCHLIST, load_prices, load_index

np.seterr(all='ignore')
t0 = time.time()
prices = load_prices(days=4000)
max_date = max(dd.index.max() for dd in prices.values())
min_date = min(dd.index.min() for dd in prices.values())
print(f"prices: {len(prices)} assets, range {min_date.date()} .. {max_date.date()} ({time.time()-t0:.1f}s)")

for s in WATCHLIST:
    df = prices[s]
    print(f"{s:10s} rows={len(df):5d} last={df.index.max().date()} last_close={df['close'].iloc[-1]:.2f} vol_ok={int(df['volume'].notna().sum())}")

print("\n--- observation signals ---")
for s in ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']:
    idx = load_index(s, days=4000, prices=prices)
    if idx is not None:
        print(f"{s:10s} rows={len(idx):5d} last={idx.index.max().date()} last_close={idx['close'].iloc[-1]:.2f}")
    else:
        print(f"{s:10s} None")

print("\n--- recent 21-session returns ---")
for s in WATCHLIST:
    df = prices[s]['close']
    r = df.iloc[-1] / df.iloc[-22] - 1.0 if len(df) > 22 else np.nan
    print(f"{s:10s} ret21={r*100:+.2f}%")

vix = load_index('VIX', days=4000, prices=prices)
if vix is not None:
    print(f"\nVIX 21d ago={vix['close'].iloc[-22]:.1f} now={vix['close'].iloc[-1]:.1f} "
          f"chg21={(vix['close'].iloc[-1]/vix['close'].iloc[-22]-1)*100:+.1f}%")
