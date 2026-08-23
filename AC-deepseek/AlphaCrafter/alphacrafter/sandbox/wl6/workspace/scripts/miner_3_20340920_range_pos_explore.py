import pandas as pd, numpy as np, os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from miner_lib import load_close_panel, cross_sectional_ic, compute_forward_rets, summarize

END = "2034-09-26"
panel, rets = load_close_panel(end=END)
print("panel shape", panel.shape, "last", panel.index.max())

def df_fact(fn):
    out = {}
    for a in panel.columns:
        out[a] = fn(a)
    return pd.DataFrame(out)

# ---------------- Range position factors ----------------
def f_range(lookback):
    def fn(a):
        d = panel[a]
        hi = d.rolling(lookback).max(); lo = d.rolling(lookback).min()
        rng = (hi-lo).replace(0, np.nan)
        return (d - lo)/rng
    return fn

def f_hi_prox(lookback):
    def fn(a):
        d = panel[a]
        hi = d.rolling(lookback).max(); lo = d.rolling(lookback).min()
        rng = (hi-lo).replace(0, np.nan)
        return (hi - d)/rng   # low value = near high = bullish
    return fn

def f_lo_sup(lookback):
    def fn(a):
        d = panel[a]
        lo = d.rolling(lookback).min()
        return (d - lo)/d   # far above recent low = bullish
    return fn

# ---------------- Acceleration / volatility factors ----------------
def f_ret_slope(short, long):
    def fn(a):
        d = panel[a]
        return d.pct_change(short) - d.pct_change(long)
    return fn

def f_vol_of_vol(lb):
    def fn(a):
        r = panel[a].pct_change()
        rv10 = r.rolling(10).std()
        return -rv10.rolling(lb).std()
    return fn

# Sharpe-of-momentum: recent return scaled by recent vol (risk-adjusted momentum)
def f_ret_over_vol(short, vlb):
    def fn(a):
        r = panel[a].pct_change()
        return r.rolling(short).sum() / (r.rolling(vlb).std())
    return fn

def report(name, fn, hor=10):
    F = df_fact(fn)
    fwd = compute_forward_rets(rets, hor)
    ics = cross_sectional_ic(F, fwd)
    s = summarize(ics, hor)
    cov = float(F.notna().mean().mean())
    Fr = F.rank(axis=1)
    to = float(Fr.diff().abs().mean().mean())
    print(f"{name:18s} ic={s['ic']:.4f} icir={s['icir']:.4f} hit={s['ic_hit_ratio']:.3f} n={s['n_ic_dates']} cov={cov:.2f} turn={to:.2f}  | {'PASS' if abs(s['ic'])>=0.0070 and abs(s['icir'])>=0.0840 else 'fail'}")
    return {**s, 'cov': cov, 'turn': to}

print("=== Horizon 10 validation through %s (15-asset cross-section) ===" % END)
print("RANGE POSITION (close position in recent high-low range)")
for lb in [10,20,60]:
    report(f"range_pos_{lb}", f_range(lb))
print("HI_PROX (distance from recent high)")
for lb in [20,60]:
    report(f"hi_prox_{lb}", f_hi_prox(lb))
print("LO_SUP (distance above recent low)")
for lb in [20,60]:
    report(f"lo_sup_{lb}", f_lo_sup(lb))
print("MOMENTUM ACCELERATION")
report("ret_slope_5_60", f_ret_slope(5,60))
report("ret_slope_10_120", f_ret_slope(10,120))
print("VOL_OF_VOL (negated)")
report("vol_of_vol_60", f_vol_of_vol(60))
report("vol_of_vol_120", f_vol_of_vol(120))
print("RISK-ADJUSTED MOMENTUM")
report("retovervol_10_60", f_ret_over_vol(10,60))
report("retovervol_20_60", f_ret_over_vol(20,60))