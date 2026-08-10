"""miner_3 2026-07-30 exploration: US10Y rate-beta conditional factor.

Idea: assets with high sensitivity (rolling 60d beta) to US10Y yield changes
tend to underperform when yields rise (rate-sensitive = duration/valuation drag)
and outperform when yields fall. Factor = asset_beta_to_US10Y * 20d change in US10Y.
Economically distinct driver (rates) vs VIX-beta (risk-off) and DXY-beta (dollar).
"""
import sys
sys.path.insert(0, 'scripts')
from factor_common import load_prices, load_index, factor_to_panel, validate_factor, build_library_panels, max_library_correlation
import json
import pandas as pd

prices = load_prices(days=2000)
us10y = load_index('US10Y')
print(f"loaded {len(prices)} assets; US10Y len={0 if us10y is None else len(us10y)}")

def us10y_beta_cond_60x20(df, s):
    if us10y is None:
        return None
    r = df['close'].pct_change()
    ry = us10y['close'].pct_change()
    z = pd.concat([r.rename('r'), ry.rename('y')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['y']) / z['y'].rolling(60).var()
    y_move = us10y['close'] / us10y['close'].shift(20) - 1.0
    return (b * y_move).reindex(z.index)

panel = factor_to_panel(us10y_beta_cond_60x20, prices)
lib = build_library_panels(prices)
m = validate_factor('us10y_beta_cond_60x20', panel, prices)
if m:
    rho, fid = max_library_correlation(panel, lib)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = fid
    print(json.dumps({k: v for k, v in m.items() if k != 'decay_ic_by_horizon'}, indent=2, default=str))
    print("decay:", json.dumps(m['decay_ic_by_horizon'], default=str))
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
    print(f"ADMISSION: |IC|={abs(m['ic']):.4f}>=0.007 {abs(m['ic'])>=0.007} | |ICIR|={abs(m['icir']):.4f}>=0.084 {abs(m['icir'])>=0.084} -> {'PASS' if ok else 'FAIL'}")
