"""Exploration sweep F (miner_3, 2026-08-16): macro-correlation change factors.

Motivation: beta_*_60 / beta_USDJPY / vix_beta_cond were explored (mostly redundant/
evicted). A SECOND-ORDER signal is the *change* in rolling correlation between an
asset and a macro anchor (DXY, USDJPY, VIX, SPX). A rising correlation with the
dollar or VIX marks regime shift; the change may predict forward 10d returns.

Candidates:
1) dxy_corr_change_60: corr(asset ret, DXY ret, 60).diff(20)  (rho change over 20d)
2) dxy_corr_change_120x60: corr(asset ret, DXY ret, 120) - corr(asset ret, DXY ret, 60)
3) vix_corr_change_60: corr(asset ret, VIX ret, 60).diff(20)
4) spx_corr_change_60: corr(asset ret, SPX ret, 60).diff(20)
5) usdjpy_corr_change_60: corr(asset ret, USDJPY ret, 60).diff(20)

Gate: abs(IC)>=0.0070 & abs(ICIR)>=0.0840 at h=10 on the 15-asset universe;
persistence additionally requires max_abs_library_correlation < 0.5.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner3_20260730_harness import ASSETS, evaluate, load_closes, load_macro  # noqa: E402

closes = load_closes()
macro = load_macro()
print("assets:", len(closes), "macro loaded:", sorted(macro.keys()))


def rolling_corr(a_ret, b_ret, n):
    return a_ret.rolling(n).corr(b_ret)


def corr_change(a_ret, b_ret, n, diff_w):
    c = rolling_corr(a_ret, b_ret, n)
    return c - c.shift(diff_w)


def corr_level_diff(a_ret, b_ret, n_long, n_short):
    return rolling_corr(a_ret, b_ret, n_long) - rolling_corr(a_ret, b_ret, n_short)


reta = {a: closes[a].pct_change() for a in closes}
dxy = macro["DXY"].pct_change()
vix = macro["VIX"].pct_change()
usdjpy = macro["USDJPY"].pct_change()
spx = closes["SPX"].pct_change()

candidates = {
    "dxy_corr_change_60x20": {a: corr_change(reta[a], dxy, 60, 20) for a in closes},
    "dxy_corr_120minus60": {a: corr_level_diff(reta[a], dxy, 120, 60) for a in closes},
    "vix_corr_change_60x20": {a: corr_change(reta[a], vix, 60, 20) for a in closes},
    "spx_corr_change_60x20": {a: corr_change(reta[a], spx, 60, 20) for a in closes},
    "usdjpy_corr_change_60x20": {a: corr_change(reta[a], usdjpy, 60, 20) for a in closes},
}

for name, vals in candidates.items():
    try:
        evaluate(closes, vals, name, horizon=10)
    except Exception as e:
        print(name, "ERROR:", repr(e))
    print()