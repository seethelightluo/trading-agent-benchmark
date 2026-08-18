"""Exploration 2 (2026-07-30): novel conditional / composite cross-asset factors.

Focus: signals complementary to existing library (mom_10d_skip5, mom_120d_skip5,
vol_of_vol20x60, vix_beta_cond_60x20) to add diversification. Use per-asset
calendars for rolling computations.

Gate: abs IC >= 0.0070 and abs ICIR >= 0.0840 at h=10 (15-instrument universe).
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_1_20260730_helpers import (WATCH, MACRO, load_panel, forward_returns,
                                      factor_ic_report, factor_turnover, coverage,
                                      decay_report, max_library_correlation)

closes = load_panel(WATCH)
rets = closes.pct_change()
macro = load_panel(MACRO)
mrets = macro.pct_change()


def per_asset(fn, panel=rets):
    out = {}
    for a in panel.columns:
        s = panel[a].dropna()
        if len(s) < 70:
            continue
        r = fn(s)
        r.index = s.index
        out[a] = r.reindex(panel.index)
    return pd.DataFrame(out)


# --- composite / normalization candidates ---
# realized vol-neutralized momentum (rank momentum over 20d divided by its own rolling std)
candidates = {}
candidates["mom20_raw_vol_adj"] = per_asset(lambda s: s.rolling(20).sum() / (s.rolling(20).std() + 1e-9))
# normalized downside semideviation
candidates["downside_ratio_20d"] = per_asset(lambda s: s.rolling(20).mean() / (np.sqrt((s.clip(upper=0) ** 2).rolling(20).mean()) + 1e-9))
# RSI-like oscillator on 14 days
candidates["rsi_14"] = per_asset(lambda s: ((s.clip(lower=0).rolling(14).mean()) /
                                            (s.clip(lower=0).rolling(14).mean() +
                                             (-s.clip(upper=0)).rolling(14).mean())))
# volatility persistence (1-day autocorr of |ret|)
candidates["vol_persist_20d"] = per_asset(lambda s: s.abs().rolling(20).apply(
    lambda v: pd.Series(v).autocorr() if len(v) > 5 else np.nan, raw=True))
# skewness ratio between upside and downside
candidates["updown_ratio_20d"] = per_asset(lambda s: (s.clip(lower=0).rolling(20).mean() + 1e-12) /
                                           ((-s.clip(upper=0)).rolling(20).mean() + 1e-12))
# conditional momentum: momentum conditioned on positive trend quality
candidates["mom20_trend_cond"] = per_asset(
    lambda s: s.rolling(20).sum() * np.where((s > 0).rolling(60).mean() > 0.5, 1.0, -1.0))
# reaction to extreme down days: 5d return after big down day (1-day shift signal)
candidates["reversal_after_drop_20d"] = per_asset(lambda s: s.rolling(20).apply(
    lambda v: float(v[-1] < np.percentile(v, 20)), raw=True))
# normalized distance from 60d high (drawdown-based mean reversion)
candidates["dist_60d_high"] = per_asset(lambda s: s / s.rolling(60).max())
# price momentum skew: difference of short and long momentum normalized
candidates["mom10_mom60_diff"] = per_asset(lambda s: (s.rolling(10).sum() - s.rolling(60).sum()) /
                                           (s.rolling(10).std() + s.rolling(60).std() + 1e-9))
# correlation-break momentum: short momentum (10d) x long quality (60d trend)
candidates["mom10_x_trend60"] = per_asset(lambda s: s.rolling(10).sum() * (s > 0).rolling(60).mean())

print(f"\n{'factor':<26}{'IC':>8}{'ICIR':>8}{'hit':>7}{'n_dates':>8}{'turn':>7}{'cov':>7}  decay1/2/3/5/10/20")
for name, f in candidates.items():
    fwd = forward_returns(rets, 10)
    rep = factor_ic_report(f, fwd, horizon=10)
    if rep is None:
        print(f"{name:<26} insufficient data")
        continue
    turn = factor_turnover(f)
    cov = coverage(f)
    dec = decay_report(f, rets)
    passed = abs(rep["ic"]) >= 0.0070 and abs(rep["icir"]) >= 0.0840
    print(f"{name:<26}{rep['ic']:>8.4f}{rep['icir']:>8.4f}{rep['ic_hit_ratio']:>7.3f}"
          f"{rep['n_ic_dates']:>8d}{turn:>7.2f}{cov['coverage_asset_days']:>7.2f}"
          f"  {dec['1']}/{dec['2']}/{dec['3']}/{dec['5']}/{dec['10']}/{dec['20']}"
          f"  {'PASS' if passed else 'fail'}")

print("\n--- library correlation for candidates with signal mass ---")
for name, f in candidates.items():
    if f.notna().sum().sum() < 500:
        continue
    max_r, detail = max_library_correlation(f)
    print(f"{name:<26} max_abs_lib_corr={max_r:.4f}")