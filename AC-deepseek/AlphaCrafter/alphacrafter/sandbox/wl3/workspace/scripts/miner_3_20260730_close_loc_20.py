"""miner_3 2026-07-30 exploration: intraday close location (20d mean).

Idea: mean over 20d of (close - low) / (high - low). Values near 1 -> closes
persistently at day highs (accumulation/buying pressure); near 0 -> closes at
lows (distribution). Uses OHLC microstructure, absent from the current library
(which is close-based or vol-based). Direction determined empirically.
"""
import sys
sys.path.insert(0, 'scripts')
from factor_common import load_prices, evaluate_candidate

prices = load_prices(days=2000)
print(f"loaded {len(prices)} assets")

def close_loc_20(df, s):
    hl = (df['high'] - df['low']).replace(0, float('nan'))
    loc = (df['close'] - df['low']) / hl
    return loc.rolling(20).mean()

m = evaluate_candidate('close_loc_20', close_loc_20, prices)
