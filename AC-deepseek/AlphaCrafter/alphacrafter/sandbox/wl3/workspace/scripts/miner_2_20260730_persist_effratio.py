"""miner_2 2026-07-30 persist batch-4 PASS factor: efficiency_ratio_20 (volume_z_20 already persisted)."""
import sys
sys.path.insert(0, 'scripts')
import json
import numpy as np
import pandas as pd
from pathlib import Path
from factor_common import (load_prices, canonical_grid, factor_to_panel,
                           WATCHLIST, persist_factor, build_library_panels)
import miner_2_20260730_screen_fast as scr

prices = load_prices(days=2000)
grid = canonical_grid(prices)

lib_mat = {}
for fid, lp in build_library_panels(prices).items():
    lib_mat[fid] = lp.reindex(grid)[WATCHLIST].values.astype(float)
for f in sorted(Path('factors').glob('*_signal.npy')):
    fid = f.name.replace('_signal.npy', '')
    if fid == 'efficiency_ratio_20':
        continue  # never persist against itself
    try:
        arr = np.load(f)
        if arr.shape == (len(grid), 15):
            lib_mat[fid] = arr.astype(float)
    except Exception:
        pass
print(f"rho audit library ({len(lib_mat)}): {sorted(lib_mat)}")


def max_rho(fac):
    best, best_id = 0.0, None
    for fid_l, lm in lib_mat.items():
        c = np.array(scr.spearman_rows(fac, lm))
        c = c[np.isfinite(c)]
        if len(c):
            r = float(np.mean(c))
            if abs(r) > best:
                best, best_id = abs(r), fid_l
    return best, best_id


def f_efficiency_ratio_20(df, s):
    close = df['close']
    net = (close - close.shift(20)).abs()
    path = close.diff().abs().rolling(20).sum()
    return (net / path.replace(0, np.nan)).reindex(close.index)


factor_id = 'efficiency_ratio_20'
panel = factor_to_panel(f_efficiency_ratio_20, prices)
m = scr.evaluate_fast(factor_id, panel)
assert m is not None
assert abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084, m
fac = panel.reindex(grid)[WATCHLIST].values.astype(float)
rho, rho_id = max_rho(fac)
assert rho < 0.5, (rho, rho_id)
m['max_abs_library_correlation'] = rho
m['max_corr_library_id'] = rho_id
m['audit_note'] = 'rho vs full persisted library (canonical recompute + npy artifacts) on 2026-07-30'

path = persist_factor(
    factor_id=factor_id,
    factor_name='Kaufman Efficiency Ratio (20d)',
    expression='|close - close_shift(20)| / sum(|diff(close)|, 20)',
    description=('Kaufman efficiency ratio: net 20-day price displacement divided by total '
                 'path length. High efficiency = smooth directional trend, low noise. Positive '
                 'IC: trend-efficient assets continue outperforming over 3-20d horizons across '
                 'the cross-asset universe (trend-quality momentum).'),
    dependencies=['close'],
    parameters={'window': 20},
    expected_direction=+1,
    panel=panel,
    metrics=m,
    tags=['trend', 'momentum', 'technical', 'cross_asset'],
    grid=grid,
    prices=prices,
    regime_notes=('2020-2026 multi-regime validation (COVID, rate shocks, crypto cycles). '
                  'Full 15/15 coverage; persistent positive IC at all horizons (h3..h20 ~ +0.04).'),
)
d = json.load(open(path))
v = d['validation']
ok = (d['factor_id'] == factor_id and v['status'] == 'EFFECTIVE'
      and abs(v['metrics']['ic']) >= 0.007 and abs(v['metrics']['icir']) >= 0.084
      and d.get('signal_artifact') and Path('factors', d['signal_artifact']).exists())
print(f"persisted {path.name}: status={v['status']} ic={v['metrics']['ic']:+.4f} "
      f"icir={v['metrics']['icir']:+.4f} rho={v['metrics']['max_abs_library_correlation']:.3f} "
      f"({v['metrics']['max_corr_library_id']}) art={d['signal_artifact']} -> {'READBACK_OK' if ok else 'READBACK_FAIL'}")
print('DONE' if ok else 'FAILED')
