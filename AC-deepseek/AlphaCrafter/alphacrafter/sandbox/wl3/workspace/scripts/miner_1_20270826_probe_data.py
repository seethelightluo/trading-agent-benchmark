"""Probe data availability as of 2027-08-26."""
import sys
sys.path.insert(0, 'scripts')
import pandas as pd
from factor_common import load_prices, load_index, WATCHLIST

prices = load_prices(days=2800)
print("assets:", len(prices))
for s in WATCHLIST:
    df = prices.get(s)
    if df is None:
        print(f"  {s}: MISSING")
    else:
        print(f"  {s}: {len(df)} rows  {df.index.min().date()}..{df.index.max().date()}  last_close={df['close'].iloc[-1]:.2f}")

for ix in ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']:
    df = load_index(ix, prices=prices)
    if df is None:
        print(f"  index {ix}: MISSING")
    else:
        print(f"  index {ix}: {len(df)} rows  {df.index.min().date()}..{df.index.max().date()}")

# max visible date across tradables
maxd = max(dd.index.max() for dd in prices.values())
print("max visible tradable date:", maxd.date())
