"""Exploration 4 (2026-07-30): focused sweep - volume, rate mean-reversion,
anchor betas not yet in library, and distance-from-high window variants.

Per-asset calendars for all rolling computations.
Gate: abs IC >= 0.0070 & abs ICIR >= 0.0840 at h=10 (15-instrument universe).
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_1_20260730_helpers import (WATCH, MACRO, load_panel, load_close, forward_returns,
                                      factor_ic_report, factor_turnover, coverage,
                                      decay_report, max_library_correlation)

closes = load_panel(WATCH)
rets = closes.pct_change()
macro = load_panel(MACRO)
mrets = macro.pct_change()
vix = mrets["VIX"]


def per_asset(fn, panel=rets, close_panel=None):
    out = {}
    for a in panel.columns:
        s = panel[a].dropna()
        if close_panel is not None:
            s = s.reindex(close_panel[a].dropna().index).dropna()
        if len(s) < 80:
            continue
        r = fn(s)
        r.index = s.index
        out[a] = r.reindex(panel.index)
    return pd.DataFrame(out)


def per_asset_c(fn):
    return per_asset(fn, panel=closes.pct_change(), close_panel=closes)


def aligned(xs_series, y):
    xs = xs_series.dropna()
    ys = y.reindex(xs.index)
    m = ys.notna()
    return xs[m], ys[m]


def beta_panel(x, y, win, self_skip=None):
    out = {}
    for a in x.columns:
        if self_skip and a == self_skip:
            out[a] = pd.Series(np.nan, index=x.index)
            continue
        xs, ys = aligned(x[a], y)
        if len(xs) < win + 5:
            out[a] = pd.Series(np.nan, index=x.index)
            continue
        df = pd.concat([xs.rename("x"), ys.rename("y")], axis=1)
        b = df["x"].rolling(win).cov(df["y"]) / df["y"].rolling(win).var()
        out[a] = b.reindex(x.index)
    return pd.DataFrame(out)


def load_vol(symbol):
    df = load_close(symbol)  # imported at top
    if df is not None and "volume" in df:
        return df["volume"].astype(float)
    return None


candidates = {}
# --- volume-based ---
vol_panel = pd.DataFrame({a: load_vol(a) for a in WATCH})
logvol = np.log(vol_panel.clip(lower=1e-9))
candidates["vol_trend_20d"] = per_asset(lambda s: s.rolling(20).mean() / (s.rolling(60).mean() + 1e-9),
                                        panel=logvol)
candidates["vol_zscore_40d"] = per_asset(lambda s: (s - s.rolling(40).mean()) / (s.rolling(40).std() + 1e-9),
                                         panel=logvol)
# Amihud-style illiquidity: |ret| / volume
illiq = rets.abs() / vol_panel.reindex(rets.index).clip(lower=1e-9)
candidates["amihud_20d"] = per_asset(lambda s: s.rolling(20).mean(), panel=illiq)

# --- rate mean reversion (yield-like assets) ---
candidates["us10y_zscore_60d"] = per_asset_c(
    lambda c: (c - c.rolling(60).mean()) / (c.rolling(60).std() + 1e-9))
candidates["price_vs_sma_60"] = per_asset_c(
    lambda c: c / c.rolling(60).mean() - 1.0)

# --- distance from high variants ---
candidates["dist_20d_high"] = per_asset_c(lambda c: c / c.rolling(20).max())
candidates["dist_120d_high"] = per_asset_c(lambda c: c / c.rolling(120).max())
candidates["dist_60d_low"] = per_asset_c(lambda c: c / c.rolling(60).min() - 1.0)
# distance from high normalized by realized vol (drawdown speed)
candidates["dist_60d_high_volnorm"] = per_asset_c(
    lambda c: (c / c.rolling(60).max()) / (c.pct_change().rolling(20).std() + 1e-9))

# --- anchor betas not yet in library ---
candidates["usdcny_beta_60d"] = beta_panel(rets, mrets["USDCNY"], 60)
candidates["eurusd_beta_60d"] = beta_panel(rets, mrets["EURUSD"], 60)
candidates["spx_beta_60d"] = beta_panel(rets, rets["SPX"], 60, self_skip="SPX")
candidates["ndx_beta_60d"] = beta_panel(rets, rets["NDX"], 60, self_skip="NDX")
# beta to VIX *conditional on vol level* (trend vs level interaction)
candidates["vix_beta_x_vix"] = (beta_panel(rets, vix, 60) *
                                vix.reindex(rets.index).rolling(20).mean()).reindex(rets.index)
# gold/oil beta spread (commodity vs energy sensitivity)
candidates["gold_oil_beta_spread"] = (beta_panel(rets, rets["XAU"], 60, self_skip="XAU") -
                                      beta_panel(rets, rets["WTI"], 60, self_skip="WTI"))

print(f"\n{'factor':<26}{'IC':>8}{'ICIR':>8}{'hit':>7}{'n_dates':>8}{'turn':>7}{'cov':>7}  decay1/2/3/5/10/20")
results = {}
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
    max_r = np.nan
    if f.notna().sum().sum() >= 500:
        max_r, _ = max_library_correlation(f)
    results[name] = dict(ic=rep["ic"], icir=rep["icir"], passed=passed, max_r=max_r)
    print(f"{name:<26}{rep['ic']:>8.4f}{rep['icir']:>8.4f}{rep['ic_hit_ratio']:>7.3f}"
          f"{rep['n_ic_dates']:>8d}{turn:>7.2f}{cov['coverage_asset_days']:>7.2f}"
          f"  {dec['1']}/{dec['2']}/{dec['3']}/{dec['5']}/{dec['10']}/{dec['20']}"
          f"  {'PASS' if passed else 'fail'}  lib={max_r:.3f}")

print("\n--- gate candidates (PASS and lib<0.6) ---")
for name, r in results.items():
    if r["passed"] and (np.isnan(r["max_r"]) or r["max_r"] < 0.6):
        print(" ", name, r)