"""miner_3 2026-07-30 exploration: Sharpe-style risk-adjusted momentum 20/60.

Idea: raw momentum can be dominated by vol; scaling 20d momentum (skip 5) by
trailing 60d vol normalizes the signal -> quality of the trend. Distinct from
library mom_10d_skip5 (pure short momentum) and mom_120d_skip5 (long raw
momentum) because it is vol-scaled at an intermediate horizon.
"""
import sys
sys.path.insert(0, 'scripts')
from factor_common import load_prices, evaluate_candidate

prices = load_prices(days=2000)
print(f"loaded {len(prices)} assets")

def vol_adj_mom_20_60(df, s):
    close = df['close']
    mom = close.shift(5) / close.shift(25) - 1.0
    vol = close.pct_change().rolling(60).std()
    return mom / vol

m = evaluate_candidate('vol_adj_mom_20_60', vol_adj_mom_20_60, prices)
