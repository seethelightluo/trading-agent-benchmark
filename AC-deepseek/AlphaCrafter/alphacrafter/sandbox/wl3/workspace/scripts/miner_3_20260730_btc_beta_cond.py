"""miner_3 2026-07-30 exploration: BTC risk-appetite conditional beta 60x20.

Idea: assets with high rolling 60d beta to BTC returns are 'risk-appetite'
exposed. When BTC trends up (risk-on), those assets should outperform; when BTC
falls (risk-off), they underperform. Factor = beta(asset, BTC, 60) * (BTC 20d ret).
Economically distinct from VIX-beta (vol fear), DXY-beta (dollar), US10Y-beta
(rates): this captures crypto-led risk appetite.
"""
import sys
sys.path.insert(0, 'scripts')
from factor_common import load_prices, factor_to_panel, validate_factor, build_library_panels, max_library_correlation
import json
import pandas as pd

prices = load_prices(days=2000)
btc = prices.get('BTC')
print(f"loaded {len(prices)} assets; BTC len={0 if btc is None else len(btc)}")

def btc_beta_cond_60x20(df, s):
    if btc is None:
        return None
    r = df['close'].pct_change()
    rb = btc['close'].pct_change()
    z = pd.concat([r.rename('r'), rb.rename('b')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['b']) / z['b'].rolling(60).var()
    btc_move = btc['close'] / btc['close'].shift(20) - 1.0
    return (b * btc_move).reindex(z.index)

panel = factor_to_panel(btc_beta_cond_60x20, prices)
print(f"Factor btc_beta_cond_60x20: panel {panel.shape} range {panel.index.min()}..{panel.index.max()}")
lib = build_library_panels(prices)
m = validate_factor('btc_beta_cond_60x20', panel, prices)
if m:
    rho, fid = max_library_correlation(panel, lib)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = fid
    print(json.dumps({k: v for k, v in m.items() if k != 'decay_ic_by_horizon'}, indent=2, default=str))
    print("decay:", json.dumps(m['decay_ic_by_horizon'], default=str))
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
    print(f"ADMISSION: |IC|={abs(m['ic']):.4f}>=0.007 {abs(m['ic'])>=0.007} | |ICIR|={abs(m['icir']):.4f}>=0.084 {abs(m['icir'])>=0.084} -> {'PASS' if ok else 'FAIL'}")
else:
    print("insufficient data -> None")
