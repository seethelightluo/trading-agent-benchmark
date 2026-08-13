"""miner_1 2032-09-02: probe data availability and recent market state."""
import sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
from factor_common import WATCHLIST, load_prices, load_index

np.seterr(all='ignore')
t0 = time.time()
prices = load_prices(days=3300)
max_date = max(dd.index.max() for dd in prices.values())
min_date = min(dd.index.min() for dd in prices.values())
print(f"prices: {len(prices)} assets, range {min_date.date()} .. {max_date.date()} ({time.time()-t0:.1f}s)")

for s in WATCHLIST:
    df = prices[s]
    print(f"{s:10s} rows={len(df):5d} last={df.index.max().date()} first={df.index.min().date()} "
          f"last_close={df['close'].iloc[-1]:.2f}")

print("\n--- observation signals ---")
for s in ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']:
    idx = load_index(s, days=3300, prices=prices)
    if idx is not None:
        print(f"{s:10s} rows={len(idx):5d} last={idx.index.max().date()} last_close={idx['close'].iloc[-1]:.2f}")
    else:
        print(f"{s:10s} None")

# recent returns
print("\n--- recent 10-session returns (last vs 10 back) ---")
for s in WATCHLIST:
    df = prices[s]['close']
    r = df.iloc[-1] / df.iloc[-11] - 1.0 if len(df) > 11 else np.nan
    print(f"{s:10s} ret10={r*100:+.2f}%")

vix = load_index('VIX', days=3300, prices=prices)
if vix is not None:
    print(f"\nVIX 20d ago={vix['close'].iloc[-21]:.1f} now={vix['close'].iloc[-1]:.1f} "
          f"chg20={(vix['close'].iloc[-1]/vix['close'].iloc[-21]-1)*100:+.1f}%")
