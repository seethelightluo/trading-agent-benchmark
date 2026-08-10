"""miner_3 2026-07-30 exploration: vol-adjusted momentum variants (longer lookbacks).

Motivation: vol_adj_mom_20_60 passed (IC=0.054, ICIR=0.160) but has rho=0.59 with
library mom_10d_skip5. Longer lookback Sharpe-style momentum should retain
predictive power while being less redundant with the short raw-momentum factor.
"""
import sys
sys.path.insert(0, 'scripts')
from factor_common import load_prices, factor_to_panel, validate_factor, build_library_panels, max_library_correlation
import json

prices = load_prices(days=2000)
lib = build_library_panels(prices)
print(f"loaded {len(prices)} assets")

def make_vam(lookback, skip, vol_win):
    def f(df, s):
        close = df['close']
        mom = close.shift(skip) / close.shift(skip + lookback) - 1.0
        vol = close.pct_change().rolling(vol_win).std()
        return mom / vol
    return f

for lb, sk, vw in [(40, 5, 60), (60, 5, 60), (60, 5, 90), (120, 5, 60), (40, 5, 90)]:
    fid = f"vam_{lb}_{vw}"
    panel = factor_to_panel(make_vam(lb, sk, vw), prices)
    m = validate_factor(fid, panel, prices)
    if m is None:
        print(f"{fid}: insufficient data -> None")
        continue
    rho, rid = max_library_correlation(panel, lib)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = rid
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
    print(f"{fid}: IC={m['ic']:.4f} ICIR={m['icir']:.4f} hit={m['ic_hit_ratio']:.3f} "
          f"rho={rho:.3f}({rid}) cov={m['coverage_asset_days']:.3f} -> {'PASS' if ok else 'FAIL'}")
