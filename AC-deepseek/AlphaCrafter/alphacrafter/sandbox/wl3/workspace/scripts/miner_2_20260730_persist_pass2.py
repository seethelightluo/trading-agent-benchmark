"""miner_2 2026-07-30 persist vol_adj_mom_20_60 + high_low_range_pos_20 (passed gate)."""
import sys
sys.path.insert(0, 'scripts')
import json
from pathlib import Path
import numpy as np
from factor_common import (load_prices, factor_to_panel, persist_factor, canonical_grid)
import miner_2_20260730_screen_fast as scr

prices = load_prices(days=2000)
grid = canonical_grid(prices)

FACTORS = [
    {
        'fid': 'vol_adj_mom_20_60',
        'name': 'Volatility-adjusted momentum 20d/60d',
        'expr': '(close.shift(5)/close.shift(25) - 1) / STD(pct_change, 60)',
        'desc': '20-day momentum (skip 5d) scaled by 60-day realized volatility. '
                'Rewards strong recent trends per unit of risk (risk-adjusted momentum). '
                'Positive IC -> direction +1. NOTE: self-reported max library correlation '
                '0.59 vs mom_10d_skip5; deterministic gate recomputes rho from artifacts.',
        'deps': ['close'],
        'params': {'mom_lookback': 20, 'skip': 5, 'vol_window': 60},
        'direction': 1,
        'tags': ['momentum', 'risk_adjusted', 'volatility'],
        'regime': '2020-2026 multi-regime; strongest at 10-20d horizon.',
    },
    {
        'fid': 'high_low_range_pos_20',
        'name': 'High-Low range position (20d)',
        'expr': '(close - min(low,20)) / (max(high,20) - min(low,20))',
        'desc': 'Position of close within the 20-day high-low range (0..1). Low values '
                '(close near range bottom) predict outperformance over 10-20d; high values '
                'predict underperformance (range mean-reversion). Direction -1 on raw value.',
        'deps': ['close', 'high', 'low'],
        'params': {'window': 20},
        'direction': -1,
        'tags': ['mean_reversion', 'technical', 'range'],
        'regime': '2020-2026 multi-regime; negative short-horizon IC flips positive by h=10.',
    },
]

for f in FACTORS:
    fid = f['fid']
    fn = dict(scr.CANDIDATES)[fid]
    panel = factor_to_panel(fn, prices)
    m = scr.evaluate_fast(fid, panel)
    assert m is not None and abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084, f"{fid}: {m}"
    metrics = {k: m[k] for k in ['ic', 'icir', 'ic_hit_ratio', 'n_ic_dates',
                                 'coverage_asset_days', 'coverage_dates_ge8',
                                 'turnover_10d_rank', 'max_abs_library_correlation',
                                 'decay_ic_by_horizon']}
    path, arr = persist_factor(
        factor_id=fid,
        factor_name=f['name'],
        expression=f['expr'],
        description=f['desc'],
        dependencies=f['deps'],
        parameters=f['params'],
        expected_direction=f['direction'],
        panel=panel,
        metrics=metrics,
        tags=f['tags'],
        grid=grid,
        prices=prices,
        version='1.0.0',
        status='EFFECTIVE',
        regime_notes=f['regime'],
    )
    print(f"PERSISTED {fid} -> {path}")
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    arr2 = np.load(Path('factors') / payload['signal_artifact'])
    checks = {
        'json_valid': True,
        'factor_id_ok': payload['factor_id'] == fid,
        'status_ok': payload['validation']['status'] == 'EFFECTIVE',
        'artifact_shape_ok': arr2.shape == (len(grid), 15),
        'artifact_matches': np.allclose(arr2, arr, equal_nan=True),
    }
    print(f"VERIFY {fid}: {checks}")
    assert all(checks.values()), f"verification failed for {fid}"
    print(f"OK {fid} persisted and reloadable, IC={metrics['ic']:.4f} ICIR={metrics['icir']:.4f} "
          f"rho_lib={metrics['max_abs_library_correlation']:.2f}")
