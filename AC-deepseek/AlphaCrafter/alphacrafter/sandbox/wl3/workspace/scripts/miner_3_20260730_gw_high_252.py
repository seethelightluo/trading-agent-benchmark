"""miner_3 2026-07-30 exploration: 52-week high proximity (George-Hwang).

Idea: distance of close below its trailing 252d high. GW show assets near their
52-week high keep outperforming (anchoring / investor inattention), while assets
far below their high stay depressed. Level-based (not return-based), distinct
from the library's return momentum factors. Direction from IC sign.
"""
import sys
sys.path.insert(0, 'scripts')
from factor_common import load_prices, evaluate_candidate

prices = load_prices(days=2000)
print(f"loaded {len(prices)} assets")

def gw_high_252(df, s):
    roll_max = df['close'].rolling(252, min_periods=60).max()
    return df['close'] / roll_max - 1.0

m = evaluate_candidate('gw_high_252', gw_high_252, prices)
