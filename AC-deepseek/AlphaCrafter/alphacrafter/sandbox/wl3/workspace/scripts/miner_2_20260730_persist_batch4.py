"""miner_2 2026-07-30 persist batch-4 PASS factors: volume_z_20, efficiency_ratio_20."""
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

# full-library rho audit (canonicals + all persisted npy artifacts)
lib_mat = {}
for fid, lp in build_library_panels(prices).items():
    lib_mat[fid] = lp.reindex(grid)[WATCHLIST].values.astype(float)
for f in sorted(Path('factors').glob('*_signal.npy')):
    fid = f.name.replace('_signal.npy', '')
    try:
        arr = np.load(f)
        if arr.shape == (len(grid), 15):
            lib_mat[fid] = arr.astype(float)
    except Exception:
        pass


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


def persist_one(factor_id, name, expr, desc, deps, params, direction, fn, tags, notes):
    panel = factor_to_panel(fn, prices)
    m = scr.evaluate_fast(factor_id, panel)
    assert m is not None, factor_id
    assert abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084, m
    fac = panel.reindex(grid)[WATCHLIST].values.astype(float)
    rho, rho_id = max_rho(fac)
    assert rho < 0.5, (factor_id, rho, rho_id)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = rho_id
    m['audit_note'] = 'rho vs full persisted library (canonical recompute + npy artifacts) on 2026-07-30'
    path = persist_factor(factor_id=factor_id, factor_name=name, expression=expr,
                          description=desc, dependencies=deps, parameters=params,
                          expected_direction=direction, panel=panel, metrics=m,
                          tags=tags, grid=grid, prices=prices,
                          regime_notes=notes)
    # read-back verification
    d = json.load(open(path))
    v = d['validation']
    ok = (d['factor_id'] == factor_id and v['status'] == 'EFFECTIVE'
          and abs(v['metrics']['ic']) >= 0.007 and abs(v['metrics']['icir']) >= 0.084
          and d.get('signal_artifact') and Path('factors') / d['signal_artifact'] in Path('factors').glob('*.npy'))
    print(f"persisted {path.name}: status={v['status']} ic={v['metrics']['ic']:+.4f} "
          f"icir={v['metrics']['icir']:+.4f} rho={v['metrics']['max_abs_library_correlation']:.3f} "
          f"({v['metrics']['max_corr_library_id']}) art={d['signal_artifact']} -> {'READBACK_OK' if ok else 'READBACK_FAIL'}")
    return ok


# ---- 1. volume_z_20 ----
def f_volume_z_20(df, s):
    v = df['volume'].replace(0, np.nan)
    return (v - v.rolling(20).mean()) / v.rolling(20).std().replace(0, np.nan)


ok1 = persist_one(
    'volume_z_20', 'Volume Z-score (20d)',
    '(volume - SMA(volume,20)) / STD(volume,20)',
    'Z-score of trading volume vs its 20-day moving average. Positive IC across the '
    'cross-asset universe: assets with abnormally expanding volume (high z) tend to '
    'outperform over the 3-10d forward horizon, consistent with volume-confirmed '
    'participation flows; strong early-horizon effect that fades by h20.',
    ['volume'], {'window': 20}, +1, f_volume_z_20,
    ['volume', 'liquidity', 'technical', 'cross_asset'],
    '2020-2026 multi-regime validation (COVID, rate shocks, crypto cycles). Available '
    'on 9/15 assets (equity indices + crypto have volume series). Decay: peaks at h5 '
    '(IC +0.058), positive from h1 through h20.')

# ---- 2. efficiency_ratio_20 ----
def f_efficiency_ratio_20(df, s):
    close = df['close']
    net = (close - close.shift(20)).abs()
    path = close.diff().abs().rolling(20).sum()
    return (net / path.replace(0, np.nan)).reindex(close.index)


ok2 = persist_one(
    'efficiency_ratio_20', 'Kaufman Efficiency Ratio (20d)',
    '|close - close_shift(20)| / sum(|diff(close)|, 20)',
    'Kaufman efficiency ratio: net 20-day price displacement divided by total path '
    'length. High efficiency = smooth directional trend, low noise. Positive IC: '
    'trend-efficient assets continue outperforming over 3-20d horizons across the '
    'cross-asset universe (trend-quality momentum).',
    ['close'], {'window': 20}, +1, f_efficiency_ratio_20,
    ['trend', 'momentum', 'technical', 'cross_asset'],
    '2020-2026 multi-regime validation (COVID, rate shocks, crypto cycles). Full 15/15 '
    'coverage; persistent positive IC at all horizons (h3..h20 ~ +0.04).')

print('ALL_OK' if (ok1 and ok2) else 'SOMETHING_FAILED')
