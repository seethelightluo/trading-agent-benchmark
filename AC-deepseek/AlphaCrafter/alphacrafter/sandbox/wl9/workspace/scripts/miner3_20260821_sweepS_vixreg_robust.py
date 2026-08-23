"""miner_3 (2026-08-21): Sweep S - robustness sweep of VIX-regime momentum.

mom_10_vixreg (close/close.shift(5)-1 signed by sign of VIX 10d change shifted
by 5) passed the gate at IC 0.0308 / ICIR 0.0899 with max lib corr 0.1021.

Now probe parameter robustness: momentum lookbacks {3,5,10,15}, VIX change
windows {5,10,20,30}, shift {0,3,5}, and level-zscore regime variant, to pick
the most stable config for persistence. Also report split-half stability.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner3_20260730_harness import ASSETS, evaluate, load_closes, load_macro

closes = load_closes()
macro = load_macro()


def build(close_map, vix_win, shift_days, mom_lb, mode="trend"):
    v = macro["VIX"].reindex(close_map["SPX"].index)
    if mode == "trend":
        vr = v.pct_change(vix_win)
    else:  # zscore
        z = (v - v.rolling(60, min_periods=30).mean()) / v.rolling(60, min_periods=30).std()
        vr = z
    signv = pd.Series(
        np.where(vr.shift(shift_days).notna(),
                 np.where(vr.shift(shift_days) > 0, -1.0, 1.0), np.nan),
        index=vr.index)
    out = {}
    for a in close_map:
        mom = close_map[a] / close_map[a].shift(mom_lb) - 1.0
        out[a] = mom * signv
    return out


variants = {
    "vixreg_w10_s5_lb5":  dict(vix_win=10, shift_days=5, mom_lb=5, mode="trend"),
    "vixreg_w10_s3_lb5":  dict(vix_win=10, shift_days=3, mom_lb=5, mode="trend"),
    "vixreg_w10_s0_lb5":  dict(vix_win=10, shift_days=0, mom_lb=5, mode="trend"),
    "vixreg_w20_s5_lb5":  dict(vix_win=20, shift_days=5, mom_lb=5, mode="trend"),
    "vixreg_w5_s5_lb5":   dict(vix_win=5, shift_days=5, mom_lb=5, mode="trend"),
    "vixreg_w30_s5_lb5":  dict(vix_win=30, shift_days=5, mom_lb=5, mode="trend"),
    "vixreg_w10_s5_lb3":  dict(vix_win=10, shift_days=5, mom_lb=3, mode="trend"),
    "vixreg_w10_s5_lb10": dict(vix_win=10, shift_days=5, mom_lb=10, mode="trend"),
    "vixreg_w10_s5_lb15": dict(vix_win=10, shift_days=5, mom_lb=15, mode="trend"),
    "vixreg_z60_s5_lb5":  dict(vix_win=10, shift_days=5, mom_lb=5, mode="zscore"),
}

results = {}
for name, kw in variants.items():
    vals = build(closes, **kw)
    try:
        res = evaluate(closes, vals, name, horizon=10)
        results[name] = res
    except Exception as e:
        print(name, "ERROR:", repr(e))
    print()

# Split-half: first half vs second half of IC series for the chosen default
print("\n=== split-half stability (vixreg_w10_s5_lb5) ===")
res = results.get("vixreg_w10_s5_lb5")
if res is not None:
    ic = res["ic_series"]
    half = len(ic) // 2
    h1, h2 = ic.iloc[:half], ic.iloc[half:]
    print(f"first half:  n={len(h1)} mean IC={h1.mean():.4f} ICIR={h1.mean()/h1.std():.4f}")
    print(f"second half: n={len(h2)} mean IC={h2.mean():.4f} ICIR={h2.mean()/h2.std():.4f}")
    # yearly breakdown
    y = ic.groupby(ic.index.year).agg(["mean", "count"])
    print("yearly IC:")
    print(y)