"""miner_3 2026-07-30 exploration: 60-day range position and range-breakout family.

Idea: hilo_pos_20d passed earlier (IC=0.032, ICIR=0.097). Longer-window range
position captures a slower trend regime, and a breakout count (fraction of last
20 days closing above the prior 60d high) measures sustained breakouts.
"""
import sys
sys.path.insert(0, 'scripts')
from factor_common import load_prices, factor_to_panel, validate_factor, build_library_panels, max_library_correlation
import numpy as np
import pandas as pd

prices = load_prices(days=2000)
lib = build_library_panels(prices)
print(f"loaded {len(prices)} assets")

def hilo_pos_60(df, s):
    hi = df['high'].rolling(60).max()
    lo = df['low'].rolling(60).min()
    return (df['close'] - lo) / (hi - lo)

def hilo_pos_120(df, s):
    hi = df['high'].rolling(120).max()
    lo = df['low'].rolling(120).min()
    return (df['close'] - lo) / (hi - lo)

def brkout_60_20(df, s):
    prior_hi = df['high'].shift(1).rolling(60).max()
    above = (df['close'] > prior_hi).astype(float)
    return above.rolling(20).mean()

cands = [('hilo_pos_60', hilo_pos_60), ('hilo_pos_120', hilo_pos_120), ('brkout_60_20', brkout_60_20)]
for fid, fn in cands:
    panel = factor_to_panel(fn, prices)
    m = validate_factor(fid, panel, prices)
    if m is None:
        print(f"{fid}: insufficient data -> None")
        continue
    rho, rid = max_library_correlation(panel, lib)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = rid
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
    print(f"{fid}: IC={m['ic']:.4f} ICIR={m['icir']:.4f} hit={m['ic_hit_ratio']:.3f} "
          f"rho={rho:.3f}({rid}) cov={m['coverage_asset_days']:.3f} n={m['n_ic_dates']} -> {'PASS' if ok else 'FAIL'}")
    print("   decay:", {h: round(v, 4) for h, v in m['decay_ic_by_horizon'].items()})
