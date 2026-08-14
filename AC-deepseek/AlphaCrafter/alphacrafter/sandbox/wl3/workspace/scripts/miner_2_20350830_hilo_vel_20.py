"""miner_2 2035-08-30 candidate: hilo_vel_20 - 20d change in 60d high-low range position.
Hypothesis: direction/velocity of recovery within the 60d range (rising from lows vs
falling from highs) carries forward-return info beyond the static level (hilo_pos_60)."""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_common import load_prices, factor_to_panel, validate_factor, canonical_grid
import miner2_common as m2

prices = load_prices(days=4200)
grid = canonical_grid(prices)

def factor_fn(df, s):
    hi = df['high'].rolling(60).max()
    lo = df['low'].rolling(60).min()
    pos = (df['close'] - lo) / (hi - lo)
    return pos - pos.shift(20)

panel = factor_to_panel(factor_fn, prices)
m = validate_factor('hilo_vel_20', panel, prices)
if m is None:
    print('hilo_vel_20: insufficient data'); sys.exit()
artifacts = m2.load_effective_artifacts()
rho, fid, details = m2.max_library_correlation(panel, artifacts, grid)
m['max_abs_library_correlation'] = rho
m['max_corr_library_id'] = fid
print(f"hilo_vel_20 panel {panel.shape} range {panel.index.min().date()}..{panel.index.max().date()}")
print(json.dumps({k: v for k, v in m.items() if k != 'decay_ic_by_horizon'}, indent=1, default=str))
print('decay:', json.dumps(m['decay_ic_by_horizon'], default=str))
ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084 and rho < 0.5
print(f"ADMISSION: |IC|={abs(m['ic']):.4f}>=0.007 {abs(m['ic'])>=0.007} | |ICIR|={abs(m['icir']):.4f}>=0.084 {abs(m['icir'])>=0.084} | rho={rho:.3f}<0.5 {rho<0.5} -> {'PASS' if ok else 'FAIL'}")
print('top corr details:', {k: round(v, 3) for k, v in sorted(details.items(), key=lambda kv: -abs(kv[1]))[:4]})
