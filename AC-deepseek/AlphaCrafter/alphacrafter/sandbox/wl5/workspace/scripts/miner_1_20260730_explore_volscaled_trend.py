"""miner_1 2026-07-30: volatility-scaled trend quality family.

Idea: the signed 30d trend R2 (trend_r2_30_signed, already EFFECTIVE) measures
how cleanly an asset trends. Dividing it by realized volatility rewards smooth
trends with low noise per unit of trend — a risk-adjusted trend-quality signal.
Variants tested: raw-vol denominator, vol-percentile denominator, and rank
products. One family per script.
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


def realized_vol(px, win):
    r = np.log(px).diff()
    return r.rolling(win, min_periods=max(int(win * 0.6), 10)).std()


def cross_rank(df, win=60):
    """Cross-sectional percentile rank with trailing rolling window."""
    out = df.rank(axis=1, pct=True)
    return out


sig30 = rolling_trend_r2(lp, 30)
vol30 = realized_vol(close, 30)
vol60 = realized_vol(close, 60)

# rank transform of vol: low vol -> high value
vol30_rank = vol30.rank(axis=1, pct=True)          # high rank = high vol
inv_vol30_rank = 1.0 - vol30_rank                   # high = low vol
vol60_rank = vol60.rank(axis=1, pct=True)
inv_vol60_rank = 1.0 - vol60_rank

variants = {
    "trend_r2_volscaled_30": sig30 / vol30,                        # raw division
    "trend_r2_x_invvolrank30": sig30 * inv_vol30_rank,             # rank product 30d vol
    "trend_r2_x_invvolrank60": sig30 * inv_vol60_rank,             # rank product 60d vol
    "trend_r2_volscaled_30_rank": sig30.div(vol30).rank(axis=1, pct=True) - 0.5,
}

results = {}
for fid, sig in variants.items():
    ic_s = ic_series(sig, fr, min_valid=8)
    n_sig = int(sig.notna().sum().sum())
    n_dates_ge8 = int(len(sig.dropna(thresh=8)))
    print(f"{fid}: IC dates={len(ic_s)}, signal cells={n_sig}, "
          f"dates>=8 assets={n_dates_ge8}/{len(sig)}")
    m = summary_metrics(ic_s, sig, fr, close, h=H)
    if m is None:
        print(f"  -> insufficient IC dates, skip")
        continue
    lib = library_ic_series_map(close, h=H)
    m["max_abs_library_correlation"] = max_abs_library_corr(ic_s, lib)
    m["regime"] = regime_split(ic_s)
    results[fid] = m
    gate = abs(m["ic"]) >= 0.007 and abs(m["icir"] or 0) >= 0.084
    print(f"=== {fid}: ic={m['ic']} icir={m['icir']} hit={m['ic_hit_ratio']} "
          f"n={m['n_ic_dates']} cov_ad={m['coverage_asset_days']} "
          f"cov_ge8={m['coverage_dates_ge8']} turn={m['turnover_10d_rank']} "
          f"max_rho_lib={m['max_abs_library_correlation']} GATE={gate}")
    print("  decay:", m["decay_ic_by_horizon"])
    print("  regimes:", m["regime"])

with open("scripts/miner_1_20260730_explore_volscaled_results.json", "w") as f:
    json.dump(results, f, indent=1, default=str)
print("\nDONE. saved results.")
