"""Family screening #1: momentum / trend / range factors on 15-asset cross-section."""
import sys, os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from miner1_common import SYMBOLS, load_close, factor_panel, ic_analysis, decay_analysis, coverage, turnover

closes = load_close()
print(f"symbols={len(closes)} dates={len(closes['SPX'])}")

def mom(nd, skip=0):
    def f(df):
        c = df["close"]
        if skip > 0:
            return c.shift(skip) / c.shift(skip + nd) - 1.0
        return c / c.shift(nd) - 1.0
    return f

def rs_mom(nd):
    def f(df):
        ret = df["close"].pct_change()
        m = ret.rolling(nd).mean() * 252
        v = ret.rolling(nd).std() * np.sqrt(252)
        return m / v
    return f

def ts_mom(nd):
    def f(df):
        ret = df["close"].pct_change()
        pos = ret.clip(lower=0).rolling(nd).sum()
        neg = (-ret.clip(upper=0)).rolling(nd).sum()
        return (pos - neg) / (pos + neg + 1e-12)
    return f

def ma_slope(n, m):
    def f(df):
        c = df["close"]
        return (c.rolling(n).mean() / c.rolling(m).mean()) - 1.0
    return f

def dist_high(nd):
    def f(df):
        return df["close"] / df["close"].rolling(nd).max() - 1.0
    return f

def dist_low(nd):
    def f(df):
        return df["close"] / df["close"].rolling(nd).min() - 1.0
    return f

def norm_ma_trend(n, m):
    def f(df):
        c = df["close"]
        ma_diff = c.rolling(n).mean() - c.rolling(m).mean()
        sd = c.rolling(n).std()
        return ma_diff / sd
    return f

cands = {
    "mom_20d": mom(20),
    "mom_60d": mom(60),
    "mom_120d": mom(120),
    "mom_250d": mom(250),
    "mom_60d_skip5": mom(60, skip=5),
    "mom_120d_skip20": mom(120, skip=20),
    "sharpe_60d": rs_mom(60),
    "ts_mom_60d": ts_mom(60),
    "ma20_60_slope": ma_slope(20, 60),
    "ma60_120_slope": ma_slope(60, 120),
    "close_vs_ma60": lambda df: df["close"] / df["close"].rolling(60).mean() - 1.0,
    "close_vs_ma120": lambda df: df["close"] / df["close"].rolling(120).mean() - 1.0,
    "dist_52w_high": dist_high(252),
    "dist_20w_high": dist_high(120),
    "dist_52w_low": dist_low(252),
    "norm_ma20_60": norm_ma_trend(20, 60),
    "norm_ma60_120": norm_ma_trend(60, 120),
}

results = {}
for name, fn in cands.items():
    panel = factor_panel(closes, fn)
    cov = coverage(panel, closes)
    to = turnover(panel)
    ic1 = ic_analysis(panel, closes, fwd_days=1)
    ic5 = ic_analysis(panel, closes, fwd_days=5)
    dec = decay_analysis(panel, closes)
    results[name] = (ic1, ic5, dec, cov)
    print(f"{name:22s} cov={cov:.3f} to={to:.3f} | IC1={ic1['ic']:+.4f} ICIR1={ic1['icir']:+.3f} hit1={ic1['hit']:.2f} n1={ic1['n_dates']} | IC5={ic5['ic']:+.4f} ICIR5={ic5['icir']:+.3f} | dec={ {k: round(v,3) for k,v in dec.items()} }")

print("\n--- gate check (|IC1|>=0.007 & |ICIR1|>=0.084) ---")
for name, (ic1, ic5, dec, cov) in results.items():
    passed = abs(ic1["ic"]) >= 0.007 and abs(ic1["icir"]) >= 0.084
    print(f"{name:22s} {'PASS' if passed else 'fail'} |IC1|={abs(ic1['ic']):.4f} |ICIR1|={abs(ic1['icir']):.3f}")