"""miner_3 2026-07-30 exploration: ETH risk-appetite conditional beta 60x20.

Idea: ETH is the most speculative crypto asset; beta(asset, ETH, 60d) times ETH
20d trend captures cross-asset exposure to crypto-led risk appetite. Distinct
from BTC-beta (more speculative beta), VIX-beta (fear), DXY-beta (dollar).
"""
import sys
sys.path.insert(0, 'scripts')
from factor_common import load_prices, factor_to_panel, validate_factor, build_library_panels, max_library_correlation
import json
import pandas as pd

prices = load_prices(days=2000)
eth = prices.get('ETH')
print(f"loaded {len(prices)} assets; ETH len={0 if eth is None else len(eth)}")

def eth_beta_cond_60x20(df, s):
    if eth is None:
        return None
    r = df['close'].pct_change()
    rb = eth['close'].pct_change()
    z = pd.concat([r.rename('r'), rb.rename('b')], axis=1, sort=False).dropna()
    b = z['r'].rolling(60).cov(z['b']) / z['b'].rolling(60).var()
    eth_move = eth['close'] / eth['close'].shift(20) - 1.0
    return (b * eth_move).reindex(z.index)

panel = factor_to_panel(eth_beta_cond_60x20, prices)
m = validate_factor('eth_beta_cond_60x20', panel, prices)
if m:
    lib = build_library_panels(prices)
    rho, fid = max_library_correlation(panel, lib)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = fid
    print(json.dumps({k: v for k, v in m.items() if k != 'decay_ic_by_horizon'}, indent=2, default=str))
    print("decay:", json.dumps(m['decay_ic_by_horizon'], default=str))
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
    print(f"ADMISSION: |IC|={abs(m['ic']):.4f}>=0.007 {abs(m['ic'])>=0.007} | |ICIR|={abs(m['icir']):.4f}>=0.084 {abs(m['icir'])>=0.084} -> {'PASS' if ok else 'FAIL'}")
else:
    print("insufficient data -> None")
