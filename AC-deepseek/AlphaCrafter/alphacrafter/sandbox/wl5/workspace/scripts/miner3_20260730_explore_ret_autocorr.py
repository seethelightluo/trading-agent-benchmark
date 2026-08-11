"""miner_3 2026-07-30: Explore lag-1 return autocorrelation factor (trend persistence vs mean-reversion).

Idea: for each asset, compute Pearson autocorrelation of daily returns with lag 1 over a trailing window.
Positive autocorr -> return shocks persist (trending); negative -> mean-reverting (choppy). This is a
path-dependence measure, distinct from net-displacement momentum (mom_10d/mom_120d) and from trend R2
(fit-quality of a straight line, which ignores the sign pattern of residuals).
"""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from miner3_lib import load_close_panel, full_validate

C, V, H, L, O = load_close_panel()
R = C.pct_change()

for win in (5, 10, 20):
    fac = pd.DataFrame(index=R.index, columns=R.columns, dtype=float)
    for s in R.columns:
        x = R[s]
        ac = x.rolling(win).apply(lambda a: a.iloc[:-1].corr(a.iloc[1:]) if len(a) >= win else np.nan, raw=False)
        fac[s] = ac
    # rank-normalize within date to make it a pure cross-sectional signal
    fac_r = fac.rank(axis=1)
    res = full_validate(fac_r, R, horizon=10, label=f"ret_autocorr_{win}")
    print("=" * 60)
    print(json.dumps({k: v for k, v in res.items() if k != 'library_rho_by_factor'}, indent=1))
    print("rho_by_factor:", json.dumps(res.get('library_rho_by_factor', {})))
    print(f"n_assets_with_data_last: {fac.iloc[-1].notna().sum()}")
