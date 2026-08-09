"""Miner3 family exploration #1: volatility / risk factors on 15-asset cross-section.
Research window capped at 2026-07-15. Volumes are not needed here.
"""
import sys, os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from miner1_common import SYMBOLS, load_close, factor_panel, ic_analysis, decay_analysis, coverage, turnover

closes = load_close()
print(f"symbols={len(closes)} dates={len(closes['SPX'])} window=2020-01-01..2026-07-15")


def vol_nd(nd):
    def f(df):
        return df["close"].pct_change().rolling(nd).std() * np.sqrt(252)
    return f


def vol_ratio(short, long):
    def f(df):
        r = df["close"].pct_change()
        return r.rolling(short).std() / r.rolling(long).std()
    return f


def downside_vol(nd):
    def f(df):
        r = df["close"].pct_change()
        neg = r.clip(upper=0)
        return neg.rolling(nd).std() * np.sqrt(252)
    return f


def upside_downside_ratio(nd):
    def f(df):
        r = df["close"].pct_change()
        up = r.clip(lower=0).rolling(nd).std()
        dn = r.clip(upper=0).rolling(nd).std()
        return up / (dn + 1e-12)
    return f


def parkinson(nd):
    def f(df):
        h = df["high"]; l = df["low"]
        hl = np.log(h / l)
        return (hl ** 2).rolling(nd).mean() / (4 * np.log(2)) * np.sqrt(252)
    return f


def range_ratio(nd):
    def f(df):
        c = df["close"]
        return (c.rolling(nd).max() - c.rolling(nd).min()) / c.rolling(nd).mean()
    return f


def vol_z(short, ref):
    def f(df):
        r = df["close"].pct_change()
        v = r.rolling(short).std()
        base = r.rolling(ref).std()
        return v / base - 1.0
    return f


def neg_freq(nd):
    def f(df):
        r = df["close"].pct_change()
        return (r < 0).rolling(nd).mean()
    return f


def var_95(nd):
    def f(df):
        r = df["close"].pct_change().rolling(nd)
        return r.quantile(0.05)
    return f


def ewma_vol(span):
    def f(df):
        r = df["close"].pct_change()
        return (r ** 2).ewm(span=span).mean().pow(0.5) * np.sqrt(252)
    return f


def max_dd(nd):
    def f(df):
        c = df["close"]
        return c.rolling(nd).max() / c - 1.0  # negative drawdown
    return f


cands = {
    "vol_20d": vol_nd(20),
    "vol_60d": vol_nd(60),
    "vol_120d": vol_nd(120),
    "vol_ratio_5_60": vol_ratio(5, 60),
    "vol_ratio_10_60": vol_ratio(10, 60),
    "vol_ratio_20_120": vol_ratio(20, 120),
    "downside_vol_20d": downside_vol(20),
    "up_dn_ratio_20d": upside_downside_ratio(20),
    "up_dn_ratio_60d": upside_downside_ratio(60),
    "parkinson_20d": parkinson(20),
    "parkinson_60d": parkinson(60),
    "range_ratio_20d": range_ratio(20),
    "range_ratio_60d": range_ratio(60),
    "vol_z_5_60": vol_z(5, 60),
    "neg_freq_60d": neg_freq(60),
    "var95_60d": var_95(60),
    "ewma_vol_20": ewma_vol(20),
    "max_dd_20d": max_dd(20),
    "max_dd_60d": max_dd(60),
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