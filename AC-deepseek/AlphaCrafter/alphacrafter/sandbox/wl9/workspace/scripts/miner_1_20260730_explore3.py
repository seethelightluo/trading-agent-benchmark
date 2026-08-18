"""Exploration 3 (2026-07-30): macro-regime-conditioned cross-asset factors.

Direction: create orthogonal signals gated on risk regime (VIX level / DXY
trend) that are complementary to the existing beta library. All rolling
computations on per-asset calendars. Report IC/ICIR at h=10 plus library rho.

Gate: abs IC >= 0.0070 & abs ICIR >= 0.0840 (15-instrument universe).
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
vix = mrets["VIX"]
dxy = mrets["DXY"]


def per_asset(fn, panel=rets):
    out = {}
    for a in panel.columns:
        s = panel[a].dropna()
        if len(s) < 80:
            continue
        r = fn(s)
        r.index = s.index
        out[a] = r.reindex(panel.index)
    return pd.DataFrame(out)


def regime(name, src, win, thresh, above):
    """Binary regime series (+1/-1) on each asset's own calendar, reindexed to full."""
    out = {}
    for a in rets.columns:
        s = rets[a].dropna()
        val = src.reindex(s.index).rolling(win).mean()
        m = val.notna()
        flag = pd.Series(np.where(m, (val > thresh) if above else (val < thresh), np.nan),
                         index=s.index)
        flag = flag.replace({0: -1.0})
        out[a] = flag.reindex(rets.index)
    return pd.DataFrame(out)


def reg_at(reg, s):
    return reg[s.name].reindex(s.index).fillna(0.0).values


vix_hi = regime("vix_hi", vix, 20, 0.0, True)   # rolling mean of VIX daily ret > 0
dxy_up = regime("dxy_up", dxy, 20, 0.0, True)

candidates = {}
# 1. short momentum gated by risk regime (defensive tilt in rising-vol regimes)
candidates["mom10_riskregime"] = per_asset(
    lambda s: s.rolling(10).sum() * np.where(reg_at(vix_hi, s) > 0, -1.0, 1.0))
# 2. own sharpe (10d) gated by USD regime
candidates["sharpe10_dxyregime"] = per_asset(
    lambda s: (s.rolling(10).mean() / (s.rolling(10).std() + 1e-9)) *
    np.where(reg_at(dxy_up, s) > 0, -1.0, 1.0))
# 3. 60d cumulative return gated by risk-off regime (macro trend tilt)
candidates["mom60_riskregime"] = per_asset(
    lambda s: s.rolling(60).sum() * np.where(reg_at(vix_hi, s) > 0, -1.0, 1.0))

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
    if f.notna().sum().sum() >= 500:
        max_r, _ = max_library_correlation(f)
        print(f"    max_abs_lib_corr={max_r:.4f}")