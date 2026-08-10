"""miner_2 2026-07-30 exploration: DXY-beta conditional 60x20.

Idea: macro currency regime factor. beta(asset_ret, DXY_ret, 60) * (DXY/DXY.shift(20)-1),
mirroring the VIX conditional factor. Positive when a USD-sensitive asset (or hedge)
aligns with the dollar trend. IC sign determined empirically.
"""
import sys
sys.path.insert(0, 'scripts')
from factor_common import (load_prices, load_index, factor_to_panel,
                           validate_factor, max_library_correlation,
                           build_library_panels)

prices = load_prices(days=2000)
dxy = load_index('DXY')
print("DXY loaded:", dxy is not None, dxy['close'].index.min() if dxy is not None else None)

def dxy_beta_cond_60x20(df, s):
    if dxy is None:
        return None
    r = df['close'].pct_change()
    dr = dxy['close'].pct_change()
    z = pd.concat([r.rename('r'), dr.rename('d')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['d']) / z['d'].rolling(60).var()
    dxy_move = dxy['close'] / dxy['close'].shift(20) - 1.0
    return (b * dxy_move).reindex(z.index)

panel = factor_to_panel(dxy_beta_cond_60x20, prices)
lib = build_library_panels(prices)
m = validate_factor('dxy_beta_cond_60x20', panel, prices)
if m:
    rho, fid = max_library_correlation(panel, lib)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = fid
    import json
    print(json.dumps({k: v for k, v in m.items() if k != 'decay_ic_by_horizon'}, indent=2, default=str))
    print("decay:", json.dumps(m['decay_ic_by_horizon'], default=str))
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
    print(f"ADMISSION: |IC|={abs(m['ic']):.4f}>=0.007 {abs(m['ic'])>=0.007} | |ICIR|={abs(m['icir']):.4f}>=0.084 {abs(m['icir'])>=0.084} -> {'PASS' if ok else 'FAIL'}")
