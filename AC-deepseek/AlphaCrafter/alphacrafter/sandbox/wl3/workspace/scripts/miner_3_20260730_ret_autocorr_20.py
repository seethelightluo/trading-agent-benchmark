"""miner_3 2026-07-30 exploration: 1-day return autocorrelation (20d window).

Idea: lag-1 autocorrelation of daily returns over trailing 20d. Positive AC ->
short-horizon momentum (continuation), negative AC -> intraday-scale reversal.
Captures a different time scale (daily microstructure) than raw 10/120d momentum.
Direction determined empirically.
"""
import sys
sys.path.insert(0, 'scripts')
from factor_common import load_prices, evaluate_candidate

prices = load_prices(days=2000)
print(f"loaded {len(prices)} assets")

def ret_autocorr_20(df, s):
    r = df['close'].pct_change()
    return r.rolling(20).corr(r.shift(1))

m = evaluate_candidate('ret_autocorr_20', ret_autocorr_20, prices)
