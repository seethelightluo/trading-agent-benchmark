"""miner_3 2026-07-30 exploration: vol term structure 20/60 ratio.

Idea: ratio of 20-day realized vol to 60-day realized vol (minus 1). Values > 0
mean short-term vol is elevated relative to the medium term (vol regime rising /
stress); values < 0 mean vol is compressing. Cross-sectionally, assets with
rising vol may be de-rated (higher discount / risk premium) or offer reversal.
Complementary to library's vol-of-vol (which is a vol level second-moment).
"""
import sys
sys.path.insert(0, 'scripts')
from factor_common import load_prices, evaluate_candidate

prices = load_prices(days=2000)
print(f"loaded {len(prices)} assets")

def vol_term_20_60(df, s):
    r = df['close'].pct_change()
    v20 = r.rolling(20).std()
    v60 = r.rolling(60).std()
    return v20 / v60 - 1.0

m = evaluate_candidate('vol_term_20_60', vol_term_20_60, prices)
