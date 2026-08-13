"""miner_3 2032-09-16 probe: verify data coverage and library state before batch-37 screen."""
import sys
sys.path.insert(0, 'scripts')
from factor_common import load_prices, WATCHLIST, VAL_START, VAL_END

prices = load_prices(days=3400)
print(f"assets loaded: {len(prices)}")
for s in WATCHLIST:
    df = prices.get(s)
    if df is None:
        print(f"  {s}: MISSING")
    else:
        print(f"  {s}: rows={len(df)} range={df.index.min().date()}..{df.index.max().date()} close_last={df['close'].iloc[-1]:.4f}")
mx = max(d.index.max() for d in prices.values())
print(f"max visible date: {mx.date()}")
print(f"VAL window: {VAL_START.date()}..{VAL_END.date()}")
