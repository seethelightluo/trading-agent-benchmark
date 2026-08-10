"""miner_1 2026-07-30: trend signal-to-noise family (price-only).

Parent trend_r2_30_signed is EFFECTIVE (ic=0.0562, icir=0.1672, cov=0.956).
Vol-scaling destroyed coverage (~0.31) due to return gaps. Instead compute the
OLS t-statistic / signal-to-noise of the trend slope from price levels alone:
    t = sqrt((n-2) * R2 / (1 - R2)) * sign(cov)
This keeps identical coverage to trend_r2_30_signed while penalising noisy
trends (low R2) relative to clean ones.
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


def rolling_trend_components(df, win):
    """Return (n, r2, signed_slope) vectorized rolling OLS of log-price ~ time."""
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
        slope = cov / var_t
        slope_sign = np.sign(cov)
    n = n.where(n >= mp)
    r2 = r2.where((var_t > 1e-12) & (var_y > 1e-12))
    slope = slope.where((var_t > 1e-12) & (var_y > 1e-12))
    return n, r2, slope, slope_sign


results = {}
for win in (30, 60):
    n, r2, slope, slope_sign = rolling_trend_components(lp, win)

    # 1. t-stat of slope
    tstat = np.sqrt((n - 2) * r2 / (1.0 - r2)) * slope_sign
    tstat = tstat.where(r2 < 0.999)  # avoid div blow-up

    # 2. snr = slope / sqrt(MSE)  <=> sqrt(R2/(1-R2)) * sqrt(n-2) scale = tstat/sqrt(n-2)
    snr = np.sqrt(r2 / (1.0 - r2)) * slope_sign

    # 3. signed slope * sqrt(n) (scale-free trend strength)
    slope_n = slope * np.sqrt(n)

    variants = {
        f"trend_tstat_{win}": tstat,
        f"trend_snr_{win}": snr,
        f"trend_slope_sqrtn_{win}": slope_n,
        f"trend_r2xs_tstat_{win}": (r2 * tstat),
    }
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

with open("scripts/miner_1_20260730_explore_trend_snr_results.json", "w") as f:
    json.dump(results, f, indent=1, default=str)
print("\nDONE. saved results.")
