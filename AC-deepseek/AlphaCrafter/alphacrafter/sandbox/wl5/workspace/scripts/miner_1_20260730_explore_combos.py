"""miner_1 2026-07-30: redundancy check + 30/60 trend combination family.

1) Quantify signal-level Spearman rank correlation of tstat/snr/slope variants
   against the persisted parent signal (signed 30d trend R2). If rho>0.9 the
   variant is a near-duplicate and must NOT be persisted as a new factor.
2) Explore 30d/60d combinations (average, acceleration, confirmation) which
   may carry incremental information and lower correlation with the parent.
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


def signed_r2(n, r2, slope_sign):
    return (slope_sign * r2).where(n.notna())


def tstat(n, r2, slope_sign):
    ts = np.sqrt((n - 2) * r2 / (1.0 - r2)) * slope_sign
    return ts.where(r2 < 0.999)


n30, r2_30, sl30, sg30 = rolling_trend_components(lp, 30)
n60, r2_60, sl60, sg60 = rolling_trend_components(lp, 60)

parent = signed_r2(n30, r2_30, sg30)
t30 = tstat(n30, r2_30, sg30)
t60 = tstat(n60, r2_60, sg60)

# --- signal-level rank correlation vs parent (mean Spearman over dates) ---
def signal_rank_corr(a, b, min_valid=8):
    rs = []
    dates = a.index.intersection(b.index)
    for d in dates:
        pair = pd.concat([a.loc[d].rename("a"), b.loc[d].rename("b")], axis=1).dropna()
        if len(pair) < min_valid:
            continue
        r = pair["a"].corr(pair["b"], method="spearman")
        if np.isfinite(r):
            rs.append(r)
    return float(np.mean(rs)) if rs else float("nan"), len(rs)

print("--- signal-level mean Spearman rank corr vs parent (signed R2 30d) ---")
cands = {"trend_tstat_30": t30, "trend_snr_30": np.sqrt(r2_30/(1-r2_30))*sg30,
         "trend_slope_sqrtn_30": sl30*np.sqrt(n30), "trend_r2xs_tstat_30": r2_30*t30,
         "trend_tstat_60": t60}
for fid, sig in cands.items():
    r, nd = signal_rank_corr(sig, parent)
    print(f"  {fid}: rank_rho_parent={r:.4f} (n={nd})")

# --- combo family ---
combos = {
    "trend_tstat_avg_30_60": 0.5 * (t30 + t60),
    "trend_tstat_accel_30_60": t30 - t60,
    "trend_confirm_30x60": t30 * (0.5 + 0.5 * sg60),       # 30d strength only when 60d agrees
    "trend_confirm_60x30": t60 * (0.5 + 0.5 * sg30),       # 60d strength only when 30d agrees
    "trend_agree_cnt": (0.5 + 0.5*sg30) + (0.5 + 0.5*sg60),  # count of agreeing horizons (0,1,2)
    "trend_tstat_30_plus_r2_60": t30 + r2_60 * sg60,       # short trend + long-run cleanliness
}
results = {}
print("\n--- combo family validation (H=10) ---")
for fid, sig in combos.items():
    ic_s = ic_series(sig, fr, min_valid=8)
    m = summary_metrics(ic_s, sig, fr, close, h=H)
    if m is None:
        print(f"{fid}: insufficient IC dates ({len(ic_s)})")
        continue
    lib = library_ic_series_map(close, h=H)
    m["max_abs_library_correlation"] = max_abs_library_corr(ic_s, lib)
    m["regime"] = regime_split(ic_s)
    rho, nd = signal_rank_corr(sig, parent)
    m["rank_rho_vs_parent"] = round(rho, 4)
    results[fid] = m
    gate = abs(m["ic"]) >= 0.007 and abs(m["icir"] or 0) >= 0.084
    print(f"=== {fid}: ic={m['ic']} icir={m['icir']} hit={m['ic_hit_ratio']} n={m['n_ic_dates']} "
          f"cov_ad={m['coverage_asset_days']} turn={m['turnover_10d_rank']} "
          f"rho_parent={rho:.4f} GATE={gate}")
    print("  decay:", m["decay_ic_by_horizon"])
    print("  regimes:", m["regime"])

with open("scripts/miner_1_20260730_explore_combos_results.json", "w") as f:
    json.dump(results, f, indent=1, default=str)
print("\nDONE. saved results.")
