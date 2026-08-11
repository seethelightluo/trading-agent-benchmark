"""miner_2 2026-07-30 — deep diagnostics for batch D passers (mom10_volreg, retvol_corr_60).
Checks: sub-period IC stability, per-asset coverage, spearman rho vs library factors
(real signal artifact panels), horizon decay, and recency (2025H2+ / 2026H1)."""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
from factor_validation_lib import (load_closes, load_index, factor_panel, validate_factor,
                                   load_library_panels, ic_series, fwd_returns,
                                   IC_GATE, ICIR_GATE)

close, vol, open_, high, low = load_closes()
macro = {
    "VIX": load_index("VIX"), "DXY": load_index("DXY"),
    "USDCNY": load_index("USDCNY"), "USDJPY": load_index("USDJPY"),
    "EURUSD": load_index("EURUSD"), "SPX": close["SPX"],
    "US10Y": close["US10Y"], "CN10Y": close["CN10Y"],
}
vix = macro["VIX"]
macro["VIX_MED60"] = vix.rolling(60).median()
lib = load_library_panels()
print("library:", list(lib.keys()))

def f_mom10_volreg(c, v, o, h, l, m):
    mom = c.shift(5) / c.shift(15) - 1.0
    r = c.pct_change()
    gate = np.where(r.rolling(20).std() < r.rolling(60).std(), 1.0, -1.0)
    return mom * pd.Series(gate, index=c.index)

def f_retvol_corr_60(c, v, o, h, l, m, win=60):
    vv = v.replace(0, np.nan)
    return c.pct_change().abs().rolling(win).corr(np.log(vv))

panels = {
    "mom10_volreg": factor_panel(f_mom10_volreg, close, vol, open_, high, low, macro),
    "retvol_corr_60": factor_panel(f_retvol_corr_60, close, vol, open_, high, low, macro),
}

REGIONS = [("2020", "2020-01-01", "2020-12-31"), ("2021", "2021-01-01", "2021-12-31"),
           ("2022", "2022-01-01", "2022-12-31"), ("2023", "2023-01-01", "2023-12-31"),
           ("2024", "2024-01-01", "2024-12-31"), ("2025H1", "2025-01-01", "2025-06-30"),
           ("2025H2", "2025-07-01", "2025-12-31"), ("2026H1", "2026-01-01", "2026-07-30")]

for name, panel in panels.items():
    print(f"\n########## {name} ##########")
    res = validate_factor({"mom10_volreg": f_mom10_volreg, "retvol_corr_60": f_retvol_corr_60}[name],
                          close, vol, open_, high, low, macro)
    print(f"  IC={res['ic']:+.4f} ICIR={res['icir']:+.4f} hit={res['ic_hit_ratio']:.3f} "
          f"n={res['n_ic_dates']} cov={res['coverage_asset_days']:.3f}/{res['coverage_dates_ge8']:.3f}")
    print(f"  decay: {res['decay_ic_by_horizon']}")
    ic10 = ic_series(panel, fwd_returns(close, 10))
    print("  sub-period IC (h=10):")
    for rn, a, b in REGIONS:
        sub = ic10.loc[(ic10.index >= a) & (ic10.index <= b)]
        if len(sub) > 0:
            print(f"    {rn:6s}: ic={sub.mean():+.4f} icir={sub.mean()/sub.std() if sub.std()>0 else np.nan:+.3f} n={len(sub)}")
    # per-asset coverage
    cov = panel.notna().mean().sort_values()
    print(f"  per-asset coverage (min->max): {dict(round(cov,2))}")
    # spearman rho vs library on common dates (pooled per-factor)
    for fid, lp in lib.items():
        common = panel.index.intersection(lp.index)
        cols = [c for c in panel.columns if c in lp.columns]
        a = panel.loc[common, cols]
        b = lp.loc[common, cols]
        rhos = []
        for col in cols:
            x, y = a[col].dropna(), b[col].dropna()
            idx = x.index.intersection(y.index)
            if len(idx) > 100:
                rhos.append(x.loc[idx].rank().corr(y.loc[idx].rank()))
        print(f"  spearman rho vs {fid}: median={np.nanmedian(rhos) if rhos else np.nan:.3f} "
              f"(n_assets={len(rhos)}) max_abs={max(abs(np.nanmedian(rhos)), 0.0):.3f}")
print("\ndone")
