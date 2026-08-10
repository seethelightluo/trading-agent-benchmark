"""miner_1 2026-07-30 cycle v6: re-validate passing trend combos + explore NEW ideas.

Candidates:
  1) trend_tstat_30_plus_r2_60  (re-validate; passed gate in v5: ic=0.0568 icir=0.1675)
  2) trend_confirm_30x60        (re-validate; passed gate in v5: ic=0.0514 icir=0.1539)
  3) risk_on_alpha_20x60        NEW: 20d momentum minus beta(60d to risk-on composite)*riskon_ret20
                                (idiosyncratic relative strength vs systematic risk)
  4) drawdown_dist_120          NEW: close/rolling_max(close,120)-1  (distance from 120d peak)
  5) accel_20x60                NEW: mom20 - mom60 (momentum acceleration)
  6) mom20_vol60                NEW: 20d momentum / 60d realized vol (risk-adjusted momentum)

All validated on the 15-asset tradable cross-asset universe, H=10, visible through 2026-07-29.
"""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_validate import (closes_panel, macro_closes, forward_returns, ic_series,
                             summary_metrics, regime_split,
                             library_ic_series_map, max_abs_library_corr)

VIS = "2026-07-29"
H = 10
close = closes_panel(VIS)
lp = np.log(close)
fr = forward_returns(close, H)
ret = close.pct_change()
print(f"panel: dates={len(close)} assets={len(close.columns)} visible_through={VIS}")


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


def tstat(n, r2, slope_sign):
    ts = np.sqrt((n - 2) * r2 / (1.0 - r2)) * slope_sign
    return ts.where(r2 < 0.999)


# ---- trend components ----
n30, r2_30, sl30, sg30 = rolling_trend_components(lp, 30)
n60, r2_60, sl60, sg60 = rolling_trend_components(lp, 60)
t30 = tstat(n30, r2_30, sg30)
t60 = tstat(n60, r2_60, sg60)

mom20 = lp.diff(20)
mom60 = lp.diff(60)
vol60 = ret.rolling(60).std() * np.sqrt(252)

# ---- risk-on composite (equal-weight of 8 equity indices) ----
EQ = ["SPX", "NDX", "SOX", "N225", "HSI", "SX5E", "000300.SH", "000688.SH"]
riskon = lp[EQ].mean(axis=1)
riskon_ret = riskon.diff()
riskon_ret20 = riskon.diff(20)

beta_riskon = {}
for a in close.columns:
    pair = pd.concat([ret[a].rename("a"), riskon_ret.rename("r")], axis=1).dropna()
    b = pair["a"].rolling(60, min_periods=36).cov(pair["r"]) / pair["r"].rolling(60, min_periods=36).var()
    beta_riskon[a] = b
beta_riskon = pd.DataFrame(beta_riskon).reindex(close.index)

cands = {
    "trend_tstat_30_plus_r2_60": t30 + r2_60 * sg60,
    "trend_confirm_30x60": t30 * (0.5 + 0.5 * sg60),
    "risk_on_alpha_20x60": (mom20 - beta_riskon * riskon_ret20),
    "drawdown_dist_120": close / close.rolling(120, min_periods=72).max() - 1.0,
    "accel_20x60": mom20 - mom60,
    "mom20_vol60": mom20 / vol60,
}

results = {}
print("\n--- validation (H=10, min 8 assets/date) ---")
for fid, sig in cands.items():
    ic_s = ic_series(sig, fr, min_valid=8)
    m = summary_metrics(ic_s, sig, fr, close, h=H)
    if m is None:
        print(f"{fid}: insufficient IC dates ({len(ic_s)}); signal cells={int(sig.notna().sum().sum())}")
        results[fid] = {"gate_pass": False, "reason": "insufficient IC dates",
                        "valid_entries": int(sig.notna().sum().sum())}
        continue
    lib = library_ic_series_map(close, h=H)
    m["max_abs_library_correlation"] = max_abs_library_corr(ic_s, lib)
    m["regime"] = regime_split(ic_s)
    results[fid] = m
    gate = abs(m["ic"]) >= 0.007 and abs(m["icir"] or 0) >= 0.084
    m["gate_pass"] = gate
    print(f"=== {fid}: ic={m['ic']} icir={m['icir']} hit={m['ic_hit_ratio']} n={m['n_ic_dates']} "
          f"cov_ad={m['coverage_asset_days']} cov_ge8={m['coverage_dates_ge8']} "
          f"turn={m['turnover_10d_rank']} max_rho_lib={m['max_abs_library_correlation']} GATE={gate}")
    print("  decay:", m["decay_ic_by_horizon"])
    print("  regimes:", m["regime"])

with open("scripts/miner_1_20260730_explore_v6_results.json", "w") as f:
    json.dump(results, f, indent=1, default=str)
print("\nDONE. saved scripts/miner_1_20260730_explore_v6_results.json")
