"""miner_3 2029-08-09 data probe: confirm price/index data availability through current date."""
import sys
sys.path.insert(0, 'scripts')
from factor_common import load_prices, load_index, WATCHLIST
import pandas as pd

prices = load_prices(days=2600)
print("assets loaded:", len(prices))
for s in WATCHLIST:
    df = prices.get(s)
    if df is not None:
        print(f"{s:10s} rows={len(df):5d} last={df.index.max().date()} first={df.index.min().date()}")
    else:
        print(f"{s:10s} MISSING")

for name in ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']:
    idx = load_index(name, prices=prices)
    if idx is not None:
        print(f"index {name:8s} rows={len(idx):5d} last={idx.index.max().date()}")
    else:
        print(f"index {name:8s} MISSING")
