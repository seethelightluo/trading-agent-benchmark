"""miner_2 2026-07-30 exploration: downside volatility ratio 60d.

Idea: volatility asymmetry (semi-deviation). Ratio of std of negative daily returns
to std of all returns over 60d. Assets with disproportionately large downside moves
(lower ratio denominator share) are crash-prone; ratio <1 means downside vol is a
large fraction of total vol. High asymmetry may predict lower forward returns.
"""
import sys
sys.path.insert(0, 'scripts')
from factor_common import load_prices, evaluate_candidate

prices = load_prices(days=2000)

def downside_vol_ratio_60(df, s):
    r = df['close'].pct_change()
    neg = r.clip(upper=0)
    sd_all = r.rolling(60).std()
    sd_neg = neg.rolling(60).std()
    return (sd_neg / sd_all).replace([float('inf')], float('nan'))

m = evaluate_candidate('downside_vol_ratio_60x20', downside_vol_ratio_60, prices)
