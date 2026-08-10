"""miner_3 2026-07-30 exploration: drawdown depth 60d.

Idea: distance of current close below the trailing 60-day high (drawdown depth).
Oversold / deeply-drawn-down assets may mean-revert (buy-the-dip) or keep
falling (falling knife). Direction determined empirically from the IC sign.
Complementary to existing library (momentum, vol-of-vol, VIX-beta).
"""
import sys
sys.path.insert(0, 'scripts')
from factor_common import load_prices, evaluate_candidate

prices = load_prices(days=2000)
print(f"loaded {len(prices)} assets")

def dd_60d(df, s):
    roll_max = df['close'].rolling(60).max()
    return df['close'] / roll_max - 1.0

m = evaluate_candidate('dd_60d', dd_60d, prices)
