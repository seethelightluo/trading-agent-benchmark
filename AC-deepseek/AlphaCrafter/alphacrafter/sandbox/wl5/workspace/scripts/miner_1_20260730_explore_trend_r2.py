"""miner_1 2026-07-30: Explore 'trend quality' factor family.

Idea: momentum works on this 15-asset cross-asset universe (mom_10d/120d passed).
Refinement: not all returns are equal - a clean, persistent linear trend (high R^2
of log-price on time) should be a stronger 10d predictor than a volatile one.
Signal = slope_sign * R^2 of OLS log(close) ~ time over window W.
Variants tested: W in {30, 60, 90}, signed (directional) and abs (consistency).
Validation via shared framework: rank IC at h=10, ICIR, hit ratio, coverage,
turnover, decay, regime splits. One idea per script.
"""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_validate import (closes_panel, forward_returns, ic_series,
                             summary_metrics, regime_split,
                             library_ic_series_map, max_abs_library_corr)

VIS = "2026-07-29"
H = 10

close = closes_panel(VIS)
ret = close.pct_change()
lp = np.log(close)
fr = forward_returns(close, H)

def trend_r2(df, win):
    """Signed R^2 of rolling OLS log-price on time (0..1), sign = slope direction."""
    t = np.arange(win, dtype=float)
    t_c = t - t.mean()
    denom = (t_c ** 2).sum()
    # rolling slope = cov(logp, t)/var(t); R^2 = corr^2
    x = df.rolling(win).apply(lambda s: np.corrcoef(s, t)[0, 1], raw=True)
    r2 = x ** 2
    # sign from end-point difference (robust)
    slope_sign = np.sign(df - df.shift(win))
    return (r2 * slope_sign).reindex(df.index)

results = {}
for win in (30, 60, 90):
    sig_signed = trend_r2(lp, win)
    m = summary_metrics(ic_series(sig_signed, fr, min_valid=8), sig_signed, fr, close, h=H)
    if m is None:
        print(f"win={win} signed: insufficient IC dates")
        continue
    lib = library_ic_series_map(close, h=H)
    m["max_abs_library_correlation"] = max_abs_library_corr(
        ic_series(sig_signed, fr, min_valid=8), lib)
    m["regime"] = regime_split(ic_series(sig_signed, fr, min_valid=8))
    fid = f"trend_r2_{win}_signed"
    results[fid] = m
    gate = abs(m["ic"]) >= 0.007 and abs(m["icir"] or 0) >= 0.084
    print(f"\n=== {fid}: ic={m['ic']} icir={m['icir']} hit={m['ic_hit_ratio']} "
          f"n={m['n_ic_dates']} cov_ad={m['coverage_asset_days']} "
          f"cov_ge8={m['coverage_dates_ge8']} turn={m['turnover_10d_rank']} "
          f"max_rho_lib={m['max_abs_library_correlation']} GATE={gate}")
    print("  decay:", m["decay_ic_by_horizon"])
    print("  regimes:", m["regime"])

    sig_abs = sig_signed.abs()
    m2 = summary_metrics(ic_series(sig_abs, fr, min_valid=8), sig_abs, fr, close, h=H)
    if m2 is not None:
        fid2 = f"trend_r2_{win}_abs"
        results[fid2] = m2
        print(f"  [{fid2}]: ic={m2['ic']} icir={m2['icir']} n={m2['n_ic_dates']}")

with open("scripts/miner_1_20260730_explore_trend_r2_results.json", "w") as f:
    json.dump(results, f, indent=1, default=str)
print("\nDONE. saved results.")
