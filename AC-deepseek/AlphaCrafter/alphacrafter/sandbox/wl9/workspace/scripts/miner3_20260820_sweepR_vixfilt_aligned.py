"""miner_3 (2026-08-20): Sweep R - VIX-regime filtered momentum, h=10 cadence aligned.

Sweep Q showed mom_10_vixreg (10d VIX change 5d ago sign-flipped x 5d momentum)
passes the IC/ICIR gate at h=10 with very low library correlation (0.1021).
Here I re-run at the aligned cadence horizon=10 (same as persistence) and probe
fresh parameter variants (VIX change windows 5/10/20, lookbacks 3/5) plus a
VIX-zscore-level variant, to pick the most robust/stable one for persistence.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner3_20260730_harness import ASSETS, evaluate, load_closes, load_macro

closes = load_closes()
macro = load_macro()

print("assets:", len(closes), "macro:", len(macro))


def mom_skip(close, lookback=5, skip=5):
    return close.shift(skip) / close.shift(skip + lookback) - 1.0


def build(close_map, vix_change_window, shift_days, lookback=5, skip=5, mode="trend"):
    v = macro["VIX"].reindex(closes["SPX"].index)
    vr = v.pct_change(vix_change_window)
    out = {}
    for a in close_map:
        m = mom_skip(close_map[a], lookback, skip)
        if mode == "trend":
            signv = pd.Series(
                np.where(vr.shift(shift_days).notna(),
                         np.where(vr.shift(shift_days) > 0, -1.0, 1.0), np.nan),
                index=vr.index)
        else:  # zscore level regime (above/below rolling median)
            z = (v - v.rolling(60, min_periods=30).mean()) / v.rolling(60, min_periods=30).std()
            signv = pd.Series(
                np.where(z.shift(shift_days).notna(),
                         np.where(z.shift(shift_days) > 0, -1.0, 1.0), np.nan),
                index=vr.index)
        out[a] = m * signv
    return out


variants = {
    "vixfilt_trend_w10_s5_lb5": dict(vix_change_window=10, shift_days=5, lookback=5, skip=5, mode="trend"),
    "vixfilt_trend_w20_s5_lb5": dict(vix_change_window=20, shift_days=5, lookback=5, skip=5, mode="trend"),
    "vixfilt_trend_w5_s3_lb5": dict(vix_change_window=5, shift_days=3, lookback=5, skip=5, mode="trend"),
    "vixfilt_trend_w10_s3_lb5": dict(vix_change_window=10, shift_days=3, lookback=5, skip=5, mode="trend"),
    "vixfilt_zsc_w60_s5_lb5": dict(vix_change_window=10, shift_days=5, lookback=5, skip=5, mode="zscore"),
    "vixfilt_trend_w10_s5_lb10": dict(vix_change_window=10, shift_days=5, lookback=10, skip=5, mode="trend"),
}

for name, kw in variants.items():
    vals = build(closes, **kw)
    try:
        evaluate(closes, vals, name, horizon=10)
    except Exception as e:
        print(name, "ERROR:", repr(e))
    print()