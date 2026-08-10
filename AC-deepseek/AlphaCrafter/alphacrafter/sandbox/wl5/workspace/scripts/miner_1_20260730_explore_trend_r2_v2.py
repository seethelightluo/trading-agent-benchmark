"""miner_1 2026-07-30: Explore 'trend quality' factor family (v2, exact vectorized).

Exact rolling OLS of log(price) ~ time using rolling sums with min_periods,
robust to NaN gaps (BTC/ETH trade on weekends, indices do not).
For window [i-win+1..i]: with t = absolute row index,
  cov = E[yt]-E[y]E[t], var_t = E[t^2]-E[t]^2, var_y = E[y^2]-E[y]^2
  slope = cov/var_t, R^2 = cov^2/(var_t*var_y), sign(slope) = sign(cov)
Slope and R^2 are invariant to constant shift of t, so absolute index is exact.
Signal variants: signed R^2 (trend quality w/ direction) and |R^2| (trendiness).
Windows 30/60/90. Validation: rank IC h=10, ICIR, hit ratio, coverage,
turnover, decay, regime split, max library rho.
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
lp = np.log(close)
fr = forward_returns(close, H)


def rolling_trend_r2(df, win):
    """Exact vectorized rolling signed R^2 of OLS log-price ~ time."""
    mp = max(int(win * 0.6), 12)
    y = df
    t = pd.Series(np.arange(len(df), dtype=float), index=df.index)
    valid = y.notna().astype(float)
    sy = y.rolling(win, min_periods=mp).sum()
    sty = y.mul(t, axis=0).rolling(win, min_periods=mp).sum()
    sy2 = (y * y).rolling(win, min_periods=mp).sum()
    st = valid.mul(t, axis=0).rolling(win, min_periods=mp).sum()
    st2 = valid.mul(t * t, axis=0).rolling(win, min_periods=mp).sum()
    n = valid.rolling(win, min_periods=mp).sum()
    with np.errstate(all="ignore"):
        mean_y = sy / n
        mean_ty = sty / n
        mean_t = st / n
        mean_y2 = sy2 / n
        mean_t2 = st2 / n
        cov = mean_ty - mean_y * mean_t
        var_t = mean_t2 - mean_t ** 2
        var_y = mean_y2 - mean_y ** 2
        r2 = (cov ** 2) / (var_t * var_y)
        slope_sign = np.sign(cov)
    sig = r2 * slope_sign
    sig = sig.where(n >= mp)
    sig = sig.where((var_t > 1e-12) & (var_y > 1e-12))
    return sig


results = {}
for win in (30, 60, 90):
    sig_signed = rolling_trend_r2(lp, win)
    ic_s = ic_series(sig_signed, fr, min_valid=8)
    n_sig = int(sig_signed.notna().sum().sum())
    n_dates_ge8 = int(len(sig_signed.dropna(thresh=8)))
    print(f"win={win} signed: IC dates={len(ic_s)}, signal cells={n_sig}, "
          f"dates>=8 assets={n_dates_ge8}/{len(sig_signed)}")
    m = summary_metrics(ic_s, sig_signed, fr, close, h=H)
    if m is None:
        print(f"win={win} signed: insufficient IC dates")
        continue
    lib = library_ic_series_map(close, h=H)
    m["max_abs_library_correlation"] = max_abs_library_corr(ic_s, lib)
    m["regime"] = regime_split(ic_s)
    fid = f"trend_r2_{win}_signed"
    results[fid] = m
    gate = abs(m["ic"]) >= 0.007 and abs(m["icir"] or 0) >= 0.084
    print(f"=== {fid}: ic={m['ic']} icir={m['icir']} hit={m['ic_hit_ratio']} "
          f"n={m['n_ic_dates']} cov_ad={m['coverage_asset_days']} "
          f"cov_ge8={m['coverage_dates_ge8']} turn={m['turnover_10d_rank']} "
          f"max_rho_lib={m['max_abs_library_correlation']} GATE={gate}")
    print("  decay:", m["decay_ic_by_horizon"])
    print("  regimes:", m["regime"])

    sig_abs = sig_signed.abs()
    ic_a = ic_series(sig_abs, fr, min_valid=8)
    m2 = summary_metrics(ic_a, sig_abs, fr, close, h=H)
    if m2 is not None:
        fid2 = f"trend_r2_{win}_abs"
        results[fid2] = m2
        print(f"  [{fid2}]: ic={m2['ic']} icir={m2['icir']} n={m2['n_ic_dates']}")

with open("scripts/miner_1_20260730_explore_trend_r2_results.json", "w") as f:
    json.dump(results, f, indent=1, default=str)
print("\nDONE. saved results.")
