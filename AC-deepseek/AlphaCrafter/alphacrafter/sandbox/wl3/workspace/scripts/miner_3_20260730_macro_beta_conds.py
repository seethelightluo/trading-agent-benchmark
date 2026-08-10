"""miner_3 2026-07-30 exploration: conditional beta family on macro drivers
NDX (tech), WTI (energy), COPPER (global growth), XAU (real rates/defensive).

Each factor = beta(asset, driver, 60d) * (driver 20d trend). Captures exposure
to distinct macro risk factors, conditional on the driver's recent direction.
"""
import sys
sys.path.insert(0, 'scripts')
from factor_common import load_prices, factor_to_panel, validate_factor, build_library_panels, max_library_correlation
import pandas as pd

prices = load_prices(days=2000)
lib = build_library_panels(prices)
print(f"loaded {len(prices)} assets")

def make_cond_beta(driver_sym, lookback=60, trend=20):
    driver = prices.get(driver_sym)
    def f(df, s):
        if driver is None:
            return None
        r = df['close'].pct_change()
        rd = driver['close'].pct_change()
        z = pd.concat([r.rename('r'), rd.rename('d')], axis=1, sort=False).dropna()
        b = z['r'].rolling(lookback).cov(z['d']) / z['d'].rolling(lookback).var()
        d_move = driver['close'] / driver['close'].shift(trend) - 1.0
        return (b * d_move).reindex(z.index)
    return f

for sym in ['NDX', 'WTI', 'COPPER', 'XAU']:
    fid = f"{sym.lower()}_beta_cond_60x20"
    panel = factor_to_panel(make_cond_beta(sym), prices)
    m = validate_factor(fid, panel, prices)
    if m is None:
        print(f"{fid}: insufficient data -> None")
        continue
    rho, rid = max_library_correlation(panel, lib)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = rid
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
    print(f"{fid}: IC={m['ic']:.4f} ICIR={m['icir']:.4f} hit={m['ic_hit_ratio']:.3f} "
          f"rho={rho:.3f}({rid}) cov={m['coverage_asset_days']:.3f} n={m['n_ic_dates']} -> {'PASS' if ok else 'FAIL'}")
