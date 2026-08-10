"""miner_2 2026-07-30 exploration: rolling skewness 60d.

Idea: return skewness. Negative skew assets are crash-prone (left tail); positive
skew assets have lottery-like upside. Cross-sectional skewness may predict
forward returns. Direction determined empirically.
"""
import sys
sys.path.insert(0, 'scripts')
from factor_common import load_prices, evaluate_candidate

prices = load_prices(days=2000)

def skew_60d(df, s):
    r = df['close'].pct_change()
    return r.rolling(60).skew()

m = evaluate_candidate('skew_60d', skew_60d, prices)
