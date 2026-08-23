"""Focused robustness/decay test for skew_20d_skip5 and variants, end 2034-09-26.

skew_20d_skip5 passed full-window gates (|IC|=0.0286>=0.007, |ICIR|=0.093>=0.084,
maxrho=0.45<0.5). Here we verify parameter robustness, horizon decay, recent
subwindow stability, and library correlation before persisting.
"""
import json
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "scripts")
import miner_shared as ms

END = "2034-09-26"
close = ms.load_close(END)
macro = ms.load_macro(END)
ret = close.pct_change()
fwd10 = ms.forward_ret(close, 10)
lib_panels = ms.library_panel(close, macro)

# Variant A: skew 20d skip5 (baseline pass)
r5 = ret.shift(5)
cands = {"skew_20d_skip5": r5.rolling(20, min_periods=12).skew(),
         "skew_20d_skip0": ret.rolling(20, min_periods=12).skew(),
         "skew_30d_skip5": r5.rolling(30, min_periods=16).skew(),
         "skew_15d_skip5": r5.rolling(15, min_periods=10).skew()}


def max_lib_corr(cand, lib_panels):
    flat = cand.stack(); best = 0.0; pairs = {}
    for name, p in lib_panels.items():
        pflat = p.reindex(cand.index).stack()
        df = pd.concat([flat.rename("f"), pflat.rename("p")], axis=1).dropna()
        if len(df) < 30:
            continue
        rho = float(df["f"].corr(df["p"]))
        pairs[name] = round(rho, 4)
        if abs(rho) > best:
            best = abs(rho)
    return best, pairs


print(f"{'variant':22s} {'IC':>8s} {'ICIR':>7s} {'hit':>5s} {'n':>5s} {'maxrho':>7s}  GATE")
for name, panel in cands.items():
    ic = ms.daily_ic(panel, fwd10)
    st = ms.ic_stats(ic, 10)
    mrho, pairs = max_lib_corr(panel, lib_panels)
    gate = "PASS" if (abs(st["ic"]) >= ms.IC_GATE and abs(st["icir"]) >= ms.ICIR_GATE and mrho < 0.5) else "fail"
    print(f"{name:24s} {st['ic']:8.4f} {st['icir']:7.3f} {st['hit']:5.2f} {st['n']:5d} {mrho:7.2f}  {gate}")
    s = ic.dropna()
    for lab, nd in (("9m", 274), ("6m", 183), ("3m", 91), ("1q30d", 30)):
        rs = s[s.index >= s.index.max() - np.timedelta64(nd, "D")]
        if len(rs):
            m = rs.mean(); sd = rs.std(ddof=1)
            print(f"    {lab}: IC {m:+.4f} ICIR {m/sd if sd>0 else float('nan'):.3f} hit {(rs>0).mean():.2f} n {len(rs)}")
    # decay across horizons
    dec = {str(h): round(ms.ic_stats(ms.daily_ic(panel, ms.forward_ret(close, h)), h)["ic"], 4)
           for h in (1, 2, 3, 5, 10, 20)}
    print("    decay:", dec)
    if name == "skew_20d_skip5":
        print("    corr_pairs:", pairs)