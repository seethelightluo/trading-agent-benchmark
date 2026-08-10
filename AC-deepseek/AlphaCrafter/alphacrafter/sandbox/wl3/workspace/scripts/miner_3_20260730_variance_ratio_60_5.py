"""miner_3 2026-07-30 exploration: variance ratio 60/5 (trend persistence).

Idea: VR = var(5d returns) / (5 * var(1d returns)) - 1 over trailing 60d.
VR > 0 -> persistent trending (random-walk variance grows faster than 1d),
VR < 0 -> mean-reverting chop. This is a scale-free trend-smoothness measure,
distinct from raw momentum (level of return) and vol-scaled momentum (return/vol).
Complementary to library mom_10d_skip5 / mom_120d_skip5 / vol_of_vol.
Direction determined empirically.
"""
import sys
sys.path.insert(0, 'scripts')
from factor_common import load_prices, evaluate_candidate

prices = load_prices(days=2000)
print(f"loaded {len(prices)} assets")

def variance_ratio_60_5(df, s):
    close = df['close']
    r1 = close.pct_change()
    r5 = close.pct_change(5)
    var1 = r1.rolling(60).var()
    var5 = r5.rolling(60).var()
    return (var5 / (5.0 * var1) - 1.0).replace([float('inf')], float('nan'))

m = evaluate_candidate('variance_ratio_60_5', variance_ratio_60_5, prices)
