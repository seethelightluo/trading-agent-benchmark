"""miner_3 2026-07-30 exploration: 20-day range position (stochastic oscillator).

Idea: (close - min(low,20)) / (max(high,20) - min(low,20)). Where the close sits
inside the recent 20-day high-low range. Near-range-high => short-term trend
continuation or overbought mean-reversion; direction empirical.
"""
import sys
sys.path.insert(0, 'scripts')
from factor_common import load_prices, evaluate_candidate

prices = load_prices(days=2000)
print(f"loaded {len(prices)} assets")

def hilo_pos_20d(df, s):
    hi = df['high'].rolling(20).max()
    lo = df['low'].rolling(20).min()
    rng = (hi - lo).replace(0, float('nan'))
    return (df['close'] - lo) / rng

m = evaluate_candidate('hilo_pos_20d', hilo_pos_20d, prices)
