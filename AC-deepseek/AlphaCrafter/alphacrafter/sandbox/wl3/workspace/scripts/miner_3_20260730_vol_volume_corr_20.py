"""miner_3 2026-07-30 exploration: return-volume correlation (20d).

Idea: rolling correlation between daily returns and volume pct_change over 20d.
Volume-confirmed moves (positive corr) indicate conviction behind direction;
negative corr -> volume fades moves. Scale-free because volume enters as pct
change, so cross-asset volume units are comparable. No existing library factor
uses volume at all -> orthogonal risk.
Direction determined empirically.
"""
import sys
sys.path.insert(0, 'scripts')
from factor_common import load_prices, evaluate_candidate

prices = load_prices(days=2000)
print(f"loaded {len(prices)} assets")

def vol_volume_corr_20(df, s):
    r = df['close'].pct_change()
    v = df['volume'].pct_change()
    return r.rolling(20).corr(v)

m = evaluate_candidate('vol_volume_corr_20', vol_volume_corr_20, prices)
