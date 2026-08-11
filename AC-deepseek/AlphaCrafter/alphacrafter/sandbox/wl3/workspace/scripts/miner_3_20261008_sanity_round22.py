"""Sanity check round22 inflated ICs using the canonical pandas validation path."""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_common import (load_prices, canonical_grid, factor_to_panel,
                           validate_factor, build_library_panels,
                           max_library_correlation, VAL_START, VAL_END)

prices = load_prices(days=2000)
grid = canonical_grid(prices)
print(f"grid {len(grid)} dates {grid.min().date()}..{grid.max().date()} | assets {len(prices)}")

lib = build_library_panels(prices)

def f_roll_kurt_60(df, s):
    return df['close'].pct_change().rolling(60).kurt()

def f_dd_depth_60(df, s):
    return df['close'] / df['close'].rolling(60).max() - 1.0

def f_weekday_effect_120(df, s):
    r = df['close'].pct_change()
    dow = r.index.dayofweek
    mon = r.where(dow == 0).rolling(120, min_periods=20).mean()
    fri = r.where(dow == 4).rolling(120, min_periods=20).mean()
    return (mon - fri)

def f_overnight_intraday_corr_20(df, s):
    gap = df['open'] / df['close'].shift(1) - 1.0
    intra = df['close'] / df['open'] - 1.0
    z = pd.concat([gap.rename('g'), intra.rename('i')], axis=1).dropna()
    return z['g'].rolling(20).corr(z['i']).reindex(z.index)

cands = {
    'roll_kurt_60': f_roll_kurt_60,
    'dd_depth_60': f_dd_depth_60,
    'weekday_effect_120': f_weekday_effect_120,
    'overnight_intraday_corr_20': f_overnight_intraday_corr_20,
}

for fid, fn in cands.items():
    panel = factor_to_panel(fn, prices)
    m = validate_factor(fid, panel, prices)
    if m is None:
        print(f"{fid}: None")
        continue
    rho, rho_id = max_library_correlation(panel, lib)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = rho_id
    print(f"\n{fid}: IC10={m['ic']:.4f} ICIR={m['icir']:.4f} hit={m['ic_hit_ratio']:.3f} "
          f"cov={m['coverage_asset_days']:.3f} ge8={m['coverage_dates_ge8']:.3f} turn={m['turnover_10d_rank']:.2f}")
    print(f"  decay={ {h: round(v,4) for h,v in m['decay_ic_by_horizon'].items()} }")
    print(f"  rho={rho:.3f} vs {rho_id}")
