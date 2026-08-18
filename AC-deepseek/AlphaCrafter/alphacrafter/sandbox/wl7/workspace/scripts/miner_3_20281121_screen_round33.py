"""miner_3 screening round 2: skew variants + tail/autocorr candidates through 2028-11-20.

Batch 2:
  G) min_ret_20d        : worst daily return over 20d (left-tail / crash sensitivity)
  H) autocorr_20d       : AR(1) of daily returns over 20d (trend persistence)
  I) gain_ratio_20d     : sum(pos ret)/sum(|ret|) over 20d (upside participation)
  J) skew_60d           : realized skewness over 60d
  K) skew_20d_skip5     : skew of returns lagged 5d (decontaminated)
  L) maxmin_20d_ratio   : max_ret - min_ret over 20d (range asymmetry)
Also reports pairwise corr of skew_20d vs library factors.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_shared import (load_close, load_macro, forward_ret, daily_ic, ic_stats,
                          rank_turnover, coverage_stats, library_panel, max_lib_corr)

END = "2028-11-20"
close = load_close(END)
macro = load_macro(END)
ret = close.pct_change()

def min_ret_20(close, window=20, min_periods=10):
    r = close.pct_change()
    return r.rolling(window, min_periods=min_periods).min()

def autocorr_20(close, window=20, min_periods=10):
    r = close.pct_change()
    def ac(x):
        if len(x) < 8:
            return np.nan
        x0 = x[:-1]; x1 = x[1:]
        if x0.std() == 0 or x1.std() == 0:
            return np.nan
        return np.corrcoef(x0, x1)[0, 1]
    return r.rolling(window, min_periods=min_periods).apply(ac, raw=True)

def gain_ratio_20(close, window=20, min_periods=10):
    r = close.pct_change()
    pos = r.clip(lower=0).rolling(window, min_periods=min_periods).sum()
    tot = r.abs().rolling(window, min_periods=min_periods).sum()
    return pos / tot

def skew_60(close, window=60, min_periods=30):
    r = close.pct_change()
    return r.rolling(window, min_periods=min_periods).skew()

def skew_20_skip5(close, window=20, skip=5, min_periods=10):
    r = close.pct_change().shift(skip)
    return r.rolling(window, min_periods=min_periods).skew()

def maxmin_ratio_20(close, window=20, min_periods=10):
    r = close.pct_change()
    mx = r.rolling(window, min_periods=min_periods).max()
    mn = r.rolling(window, min_periods=min_periods).min()
    return mx - mn

CAND = {
    "min_ret_20d": min_ret_20(close),
    "autocorr_20d": autocorr_20(close),
    "gain_ratio_20d": gain_ratio_20(close),
    "skew_60d": skew_60(close),
    "skew_20d_skip5": skew_20_skip5(close),
    "maxmin_20d_ratio": maxmin_ratio_20(close),
}

lib_panels = library_panel(close, macro)
fwd10 = forward_ret(close, 10)

print(f"{'factor':22s} {'IC':>8s} {'ICIR':>7s} {'hit':>6s} {'n':>6s} {'turn10':>7s} {'cov':>6s} {'max|rho|':>8s}")
for name, f in CAND.items():
    ic = daily_ic(f, fwd10)
    st = ic_stats(ic, 10)
    turn = rank_turnover(f, 10)
    cov = coverage_stats(f, fwd10)
    mlc, pairs = max_lib_corr(f, lib_panels)
    print(f"{name:22s} {st['ic']:8.4f} {st['icir']:7.3f} {st['hit']:6.3f} {st['n']:6d} "
          f"{turn:7.3f} {cov['coverage_asset_days']:6.3f} {mlc:8.3f}")

# skew_20d pairwise vs library
print("\nskew_20d pairwise corr vs library:")
f = skew_20(close) if False else None
r = close.pct_change()
sk = r.rolling(20, min_periods=10).skew()
flat = sk.stack()
for name, p in lib_panels.items():
    pflat = p.reindex(sk.index).stack()
    df = pd.concat([flat.rename("f"), pflat.rename("p")], axis=1).dropna()
    if len(df) > 30:
        print(f"  {name:24s} rho={df['f'].corr(df['p']):+.4f}")

print("\nPer-year h10 IC for skew_20d_skip5 and min_ret_20d:")
for name in ["skew_20d_skip5", "min_ret_20d"]:
    f = CAND[name]
    ic = daily_ic(f, fwd10)
    yrs = {}
    for y in range(2020, 2029):
        sub = ic[ic.index.year == y].dropna()
        if len(sub) > 30:
            yrs[y] = (sub.mean(), len(sub))
    s = " ".join(f"{y}:{v[0]:+.4f}({v[1]})" for y, v in sorted(yrs.items()))
    print(f"{name:22s} {s}")
