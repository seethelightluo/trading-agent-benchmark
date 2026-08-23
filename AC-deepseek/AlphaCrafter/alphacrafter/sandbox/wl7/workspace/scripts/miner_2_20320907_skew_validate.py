"""miner_2 focused validation of skew_20d_skip5 (visible 2032-09-03).
Per-year IC/ICIR, decay by horizon, coverage, turnover, max lib correlation.
"""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np, pandas as pd
from miner_shared import (load_close, load_macro, forward_ret, daily_ic, ic_stats,
                          coverage_stats, library_panel, max_lib_corr, rank_turnover, summarize)

END = "2032-09-03"
close = load_close(END); macro = load_macro(END); lib = library_panel(close, macro)
ret = close.pct_change(); fwd = forward_ret(close, 10)
r = ret.shift(5)
f = r.rolling(20, min_periods=12).skew()

def per_year(ic):
    s = ic.dropna()
    df = pd.DataFrame({"ic": s}); df["year"] = s.index.year
    out = []
    for y, g in df.groupby("year"):
        m = g["ic"].mean(); sd = g["ic"].std(ddof=1)
        out.append((y, float(m), float(m/sd) if sd>0 else np.nan, int(len(g))))
    return out

summ = summarize(f, close)
print("== decay by horizon ==")
for h, st in summ.items():
    print(f"  h{h}: IC {st['ic']:+.4f}  ICIR {st['icir']:+.3f}  hit {st['hit']:.2f}  n {st['n']}")

ic = daily_ic(f, fwd)
st = ic_stats(ic, 10)
cov = coverage_stats(f, fwd)
turn = rank_turnover(f, 10)
mrho, pairs = max_lib_corr(f, lib)
print(f"\nh10 full: IC {st['ic']:+.4f}  ICIR {st['icir']:+.3f}  hit {st['hit']:.3f}  n {st['n']}")
print(f"cover_asset_days {cov['coverage_asset_days']:.3f}  dates_ge8 {cov['coverage_dates_ge8']:.3f}  turnover10 {turn:.3f}")
print(f"max_abs_lib_corr {mrho:.4f}  pairs {pairs}")
print("\n== per-year ==")
for y, m, ir, n in per_year(ic):
    print(f"  {y}: IC {m:+.4f}  ICIR {ir:+.3f}  n {n}")
print("\n== recent 2y / 1y / recent 6m IC stats ==")
for lbl, dt in [("2y","2030-09-01"),("1y","2031-09-01"),("6m","2032-03-01")]:
    s = ic.dropna(); s = s[s.index >= dt]
    if len(s):
        m = s.mean(); sd = s.std(ddof=1)
        print(f"  {lbl}: IC {m:+.4f}  ICIR {m/sd if sd>0 else np.nan:+.3f}  hit {(s>0).mean():.2f}  n {len(s)}")