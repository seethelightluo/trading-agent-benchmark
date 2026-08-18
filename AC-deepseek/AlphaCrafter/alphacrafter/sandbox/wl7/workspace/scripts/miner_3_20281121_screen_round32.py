"""miner_3 screening round: novel factor candidates through 2028-11-20.

Candidates (h10 admission gate |IC|>=0.0070, |ICIR|>=0.0840):
  A) skew_20d          : realized skewness of daily returns (lottery demand)
  B) drawdown_60d      : 1 - close/rolling_max(close,60) (depth below high)
  C) sharpe_20d        : mean/std of daily returns over 20d (risk-adj momentum)
  D) usdjpy_beta_cond  : beta to USDJPY * USDJPY 20d mom (carry unwind proxy)
  E) wti_beta_cond     : beta to WTI * WTI 20d mom (energy beta proxy)
  F) range_20d         : mean(high-low)/close over 20d (range vol)
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_shared import (load_close, load_macro, forward_ret, daily_ic, ic_stats,
                          summarize, rank_turnover, coverage_stats, library_panel,
                          max_lib_corr, ASSETS)

END = "2028-11-20"
close = load_close(END)
macro = load_macro(END)
ret = close.pct_change()

# ---- A) skew_20d ----
def skew_20(close, window=20, min_periods=10):
    r = close.pct_change()
    return r.rolling(window, min_periods=min_periods).skew()

# ---- B) drawdown_60d ----
def drawdown_60(close, window=60):
    return 1.0 - close / close.rolling(window, min_periods=30).max()

# ---- C) sharpe_20d ----
def sharpe_20(close, window=20, min_periods=10):
    r = close.pct_change()
    m = r.rolling(window, min_periods=min_periods).mean()
    s = r.rolling(window, min_periods=min_periods).std()
    return m / s

# ---- D) usdjpy_beta_cond_60x20 ----
def usdjpy_beta_cond(close, usdjpy, beta_win=60, cond_win=20, min_periods=30):
    r = close.pct_change()
    fx = usdjpy.pct_change()
    cov = r.rolling(beta_win, min_periods=min_periods).cov(fx)
    var = fx.rolling(beta_win, min_periods=min_periods).var()
    beta = cov.divide(var, axis=0)
    fx_mom = usdjpy / usdjpy.shift(cond_win) - 1.0
    return beta.multiply(fx_mom, axis=0)

# ---- E) wti_beta_cond_60x20 ----
def wti_beta_cond(close, wti, beta_win=60, cond_win=20, min_periods=30):
    r = close.pct_change()
    w = wti.pct_change()
    cov = r.rolling(beta_win, min_periods=min_periods).cov(w)
    var = w.rolling(beta_win, min_periods=min_periods).var()
    beta = cov.divide(var, axis=0)
    w_mom = wti / wti.shift(cond_win) - 1.0
    return beta.multiply(w_mom, axis=0)

# ---- F) range_20d ----
def range_20(close, window=20, min_periods=10):
    hi = pd.concat([close[a] for a in close.columns], axis=1)
    lo = pd.concat([close[a] for a in close.columns], axis=1)
    # use daily range proxy via high-low is not available; use abs ret range instead
    r = close.pct_change().abs()
    return r.rolling(window, min_periods=min_periods).mean()

CAND = {
    "skew_20d": skew_20(close),
    "drawdown_60d": drawdown_60(close),
    "sharpe_20d": sharpe_20(close),
    "usdjpy_beta_cond_60x20": usdjpy_beta_cond(close, macro["USDJPY"]),
    "wti_beta_cond_60x20": wti_beta_cond(close, close["WTI"]),
    "range_20d": range_20(close),
}

lib_panels = library_panel(close, macro)
fwd10 = forward_ret(close, 10)

print(f"{'factor':26s} {'IC':>8s} {'ICIR':>7s} {'hit':>6s} {'n':>6s} {'turn10':>7s} {'cov':>6s} {'max|rho|':>8s}")
for name, f in CAND.items():
    ic = daily_ic(f, fwd10)
    st = ic_stats(ic, 10)
    turn = rank_turnover(f, 10)
    cov = coverage_stats(f, fwd10)
    mlc, pairs = max_lib_corr(f, lib_panels)
    print(f"{name:26s} {st['ic']:8.4f} {st['icir']:7.3f} {st['hit']:6.3f} {st['n']:6d} "
          f"{turn:7.3f} {cov['coverage_asset_days']:6.3f} {mlc:8.3f}")

print("\nPer-year h10 IC (direction-agnostic |IC| used for gate):")
for name, f in CAND.items():
    ic = daily_ic(f, fwd10)
    yrs = {}
    for y in range(2020, 2029):
        sub = ic[ic.index.year == y].dropna()
        if len(sub) > 30:
            yrs[y] = (sub.mean(), len(sub))
    s = " ".join(f"{y}:{v[0]:+.4f}({v[1]})" for y, v in sorted(yrs.items()))
    print(f"{name:26s} {s}")
