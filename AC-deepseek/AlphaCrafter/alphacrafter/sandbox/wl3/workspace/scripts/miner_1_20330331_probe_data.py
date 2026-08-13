"""miner_1 2033-03-31: probe data availability through visible 2033-03-30."""
import sys
sys.path.insert(0, 'scripts')
from factor_common import WATCHLIST, load_prices, load_index

prices = load_prices(days=3400)
max_date = max(dd.index.max() for dd in prices.values())
print(f"watchlist: {len(prices)} assets, last date {max_date.date()}")
for s in WATCHLIST:
    dd = prices.get(s)
    if dd is None:
        print(f"{s:10s} MISSING")
        continue
    print(f"{s:10s} rows={len(dd):5d} first={dd.index.min().date()} last={dd.index.max().date()}")

print("--- observation-only (index_data csv) ---")
for s in ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']:
    try:
        df = load_index(s)
        print(f"{s:10s} rows={len(df):5d} first={df.index.min().date()} last={df.index.max().date()}")
    except Exception as e:
        print(f"{s:10s} ERR {e}")
