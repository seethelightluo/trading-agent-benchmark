"""Exploration: batch of candidate cross-asset factors, IC/ICIR at h=10.

Mixed calendars (BTC/ETH trade 7d/wk; other assets on their own calendars): all
rolling computations run on per-asset dropna series; macro signals (DXY, VIX)
are aligned on each asset's own dates before rolling beta/correlation.

Admission gate (15-instrument universe): abs daily paper IC >= 0.0070 and
abs daily paper ICIR >= 0.0840 at horizon h=10.
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
print(f"closes shape: {closes.shape}  range: {closes.index.min().date()} -> {closes.index.max().date()}")
print(f"n assets: {closes.shape[1]}  macro cols: {list(macro.columns)}")


def per_asset(fn, panel=rets):
    out = {}
    for a in panel.columns:
        s = panel[a].dropna()
        if len(s) < 70:
            out[a] = pd.Series(np.nan, index=panel.index)
            continue
        out[a] = fn(s).reindex(panel.index)
    return pd.DataFrame(out)


def per_asset_close(fn):
    out = {}
    for a in closes.columns:
        c = closes[a].dropna()
        if len(c) < 70:
            out[a] = pd.Series(np.nan, index=closes.index)
            continue
        out[a] = fn(c).reindex(closes.index)
    return pd.DataFrame(out)


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


def corr_change_panel(x, y, w1, w2):
    out = {}
    for a in x.columns:
        xs, ys = aligned(x[a], y)
        if len(xs) < w2 + 5:
            out[a] = pd.Series(np.nan, index=x.index)
            continue
        df = pd.concat([xs.rename("x"), ys.rename("y")], axis=1)
        out[a] = (df["x"].rolling(w1).corr(df["y"]) -
                  df["x"].rolling(w2).corr(df["y"])).reindex(x.index)
    return pd.DataFrame(out)


def max_dd_fn(v):
    try:
        c = np.cumprod(np.r_[1.0, v])
        dd = c / np.maximum.accumulate(c) - 1.0
        return dd.min()
    except Exception:
        return np.nan


def acf(v):
    if len(v) < 5:
        return np.nan
    return pd.Series(v).autocorr()


candidates = {}
# --- macro / cross-asset beta family ---
candidates["dxy_beta_60d"] = beta_panel(rets, mrets["DXY"], 60)
candidates["rate_beta_60d"] = beta_panel(rets, rets["US10Y"], 60, self_skip="US10Y")
candidates["vix_beta_plain_60d"] = beta_panel(rets, mrets["VIX"], 60)
candidates["crypto_beta_60d"] = beta_panel(rets, rets["BTC"], 60, self_skip="BTC")
candidates["oil_beta_60d"] = beta_panel(rets, rets["WTI"], 60, self_skip="WTI")
candidates["gold_beta_60d"] = beta_panel(rets, rets["XAU"], 60, self_skip="XAU")
candidates["dxy_corr_change_20_60"] = corr_change_panel(rets, mrets["DXY"], 20, 60)
# --- distribution / risk family ---
candidates["skew_20d"] = per_asset(lambda s: s.rolling(20).skew())
candidates["vol_ratio_5_60"] = per_asset(lambda s: s.rolling(5).std() / s.rolling(60).std())
candidates["downside_vol_60d"] = per_asset(lambda s: np.sqrt((s.clip(upper=0) ** 2).rolling(60).mean()))
candidates["trend_quality_60d"] = per_asset(lambda s: (s > 0).rolling(60).mean())
candidates["autocorr_10d"] = per_asset(lambda s: s.rolling(10).apply(acf, raw=True))
candidates["max_drawdown_60d"] = per_asset(lambda s: s.rolling(60).apply(max_dd_fn, raw=True))
candidates["downside_ratio_60d"] = per_asset(lambda s: s.rolling(60).mean() / np.sqrt((s.clip(upper=0) ** 2).rolling(60).mean()))
# --- trend / momentum family ---
candidates["mom60_vol_adj"] = per_asset(lambda s: s.rolling(60).sum() / s.rolling(60).std())
candidates["mom_20_60_spread"] = per_asset(lambda s: s.rolling(20).sum() - s.rolling(60).sum())
candidates["ret_reversal_1d"] = per_asset(lambda s: -s)
candidates["range_pos_20d"] = per_asset_close(lambda c: (c - c.rolling(20).min()) /
                                              (c.rolling(20).max() - c.rolling(20).min()))

print(f"\n{'factor':<26}{'IC':>8}{'ICIR':>8}{'hit':>7}{'n_dates':>8}{'turn':>7}{'cov':>7}  decay1/2/3/5/10/20")
for name, f in candidates.items():
    h = 10
    fwd = forward_returns(rets, h)
    rep = factor_ic_report(f, fwd, horizon=h)
    if rep is None:
        print(f"{name:<26} insufficient data (no IC dates with >=8 valid)")
        continue
    turn = factor_turnover(f)
    cov = coverage(f)
    dec = decay_report(f, rets)
    passed = abs(rep["ic"]) >= 0.0070 and abs(rep["icir"]) >= 0.0840
    print(f"{name:<26}{rep['ic']:>8.4f}{rep['icir']:>8.4f}{rep['ic_hit_ratio']:>7.3f}"
          f"{rep['n_ic_dates']:>8d}{turn:>7.2f}{cov['coverage_asset_days']:>7.2f}"
          f"  {dec['1']}/{dec['2']}/{dec['3']}/{dec['5']}/{dec['10']}/{dec['20']}"
          f"  {'PASS' if passed else 'fail'}")

print("\n--- library correlation for near-gate candidates (provenance) ---")
for name, f in candidates.items():
    if f.notna().sum().sum() < 500:
        continue
    max_r, detail = max_library_correlation(f)
    print(f"{name:<26} max_abs_lib_corr={max_r:.4f}  {detail}")