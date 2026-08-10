"""miner_1 2026-07-30 cycle v6b: fixed/expanded NEW candidate exploration.

Fixes vs v6:
  - risk_on_alpha: use .mul(axis=0) for Series-on-DataFrame alignment (was column-align -> all NaN)
  - vol-based candidates: min_periods=36 (union calendar has weekend NaNs for index assets)

New candidates:
  1) risk_on_alpha_20x60       20d momentum minus 60d risk-on beta * riskon_ret20
  2) mom20_vol60               20d momentum / 60d realized vol (min_periods=36)
  3) trend_tstat_30_div_vol20  trend t-stat (30d) divided by 20d vol (risk-adjusted trend)
  4) downside_adj_mom20_60     20d momentum / 60d downside deviation
  5) vix_state_mom_20x120      20d momentum only when VIX below its 120d median (low-stress regime)
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


n30, r2_30, sl30, sg30 = rolling_trend_components(lp, 30)
t30 = tstat(n30, r2_30, sg30)

mom20 = lp.diff(20)
riskon = lp[["SPX", "NDX", "SOX", "N225", "HSI", "SX5E", "000300.SH", "000688.SH"]].mean(axis=1)
riskon_ret = riskon.diff()
riskon_ret20 = riskon.diff(20)

beta_riskon = {}
for a in close.columns:
    pair = pd.concat([ret[a].rename("a"), riskon_ret.rename("r")], axis=1).dropna()
    b = pair["a"].rolling(60, min_periods=36).cov(pair["r"]) / pair["r"].rolling(60, min_periods=36).var()
    beta_riskon[a] = b
beta_riskon = pd.DataFrame(beta_riskon).reindex(close.index)

vol20 = ret.rolling(20, min_periods=12).std() * np.sqrt(252)
vol60 = ret.rolling(60, min_periods=36).std() * np.sqrt(252)
down = ret.where(ret < 0, 0.0)
downside_dev60 = np.sqrt((down ** 2).rolling(60, min_periods=36).mean()) * np.sqrt(252)

vix = macro_closes(VIS)["VIX"]
vix_low = (vix < vix.rolling(120, min_periods=72).median()).astype(float).reindex(close.index).ffill()

cands = {
    "risk_on_alpha_20x60": mom20 - beta_riskon.mul(riskon_ret20, axis=0),
    "mom20_vol60": mom20 / vol60,
    "trend_tstat_30_div_vol20": t30 / vol20,
    "downside_adj_mom20_60": mom20 / downside_dev60,
    "vix_state_mom_20x120": mom20.mul(vix_low, axis=0),
}

results = {}
print(f"panel: dates={len(close)} assets={len(close.columns)} visible_through={VIS}\n")
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

with open("scripts/miner_1_20260730_explore_v6b_results.json", "w") as f:
    json.dump(results, f, indent=1, default=str)
print("\nDONE. saved scripts/miner_1_20260730_explore_v6b_results.json")
