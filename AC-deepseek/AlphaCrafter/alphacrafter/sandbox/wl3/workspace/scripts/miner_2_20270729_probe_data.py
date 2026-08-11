"""Probe data availability as of current sim date (2027-07-29)."""
import sys
sys.path.insert(0, 'scripts')
from factor_common import load_prices, load_index, WATCHLIST, INDEX_SIGNALS

prices = load_prices(days=4000)
for s in WATCHLIST:
    df = prices.get(s)
    if df is None:
        print(f"{s}: NO DATA")
    else:
        print(f"{s}: rows={len(df)} range={df.index.min().date()}..{df.index.max().date()}")

print("--- index signals ---")
for s in INDEX_SIGNALS:
    df = load_index(s, days=4000, prices=prices)
    if df is None:
        print(f"{s}: NO DATA")
    else:
        print(f"{s}: rows={len(df)} range={df.index.min().date()}..{df.index.max().date()} last={df['close'].iloc[-1]:.2f}")
