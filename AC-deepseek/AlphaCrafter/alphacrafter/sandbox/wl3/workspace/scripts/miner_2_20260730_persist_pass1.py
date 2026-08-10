"""miner_2 2026-07-30 persist passing candidates (rsi_14d, bollinger_z_20d).

Writes factors/<fid>.json + factors/<fid>_signal.npy on the canonical grid so
the deterministic gate can recompute pairwise rho.
"""
import sys
sys.path.insert(0, 'scripts')
import json
from pathlib import Path
import numpy as np
import pandas as pd
from factor_common import (load_prices, load_index, factor_to_panel, persist_factor,
                           VAL_START, VAL_END, canonical_grid)
import miner_2_20260730_screen_fast as scr

prices = load_prices(days=2000)
grid = canonical_grid(prices)

FACTORS = [
    {
        'fid': 'rsi_14d',
        'name': 'RSI(14) mean-reversion',
        'expr': '100 - 100/(1 + EWM(|up|,14)/EWM(|down|,14))',
        'desc': 'Wilder-style RSI-14 on daily closes. High RSI (overbought) assets are '
                'expected to underperform over ~2-week horizon, low RSI to outperform '
                '(cross-asset mean reversion). Admission at h=10: positive IC = low RSI '
                'wins, i.e. direction=-1 on raw RSI.',
        'deps': ['close'],
        'params': {'lookback': 14, 'smoothing': 'ewm_alpha_1/14'},
        'direction': -1,
        'tags': ['mean_reversion', 'momentum', 'oscillator'],
        'regime': '2020-2026 includes COVID crash, 2022 bear, 2023-2025 crypto/equity bull; '
                  'works as cross-asset contrarian signal.',
    },
    {
        'fid': 'bollinger_z_20d',
        'name': 'Bollinger Z-score (20d)',
        'expr': '(close - SMA(close,20)) / STD(close,20)',
        'desc': 'Z-score of close vs 20-day moving average. Positive z (price stretched '
                'above band) predicts underperformance over 10-20d horizon; negative z '
                'predicts outperformance (cross-asset band mean-reversion).',
        'deps': ['close'],
        'params': {'window': 20, 'std_ddof': 1},
        'direction': -1,
        'tags': ['mean_reversion', 'volatility', 'technical'],
        'regime': '2020-2026 multi-regime validation (COVID, rate shocks, crypto cycles).',
    },
]

for f in FACTORS:
    fid = f['fid']
    fn = dict(scr.CANDIDATES)[fid]
    panel = factor_to_panel(fn, prices)
    m = scr.evaluate_fast(fid, panel)
    if m is None:
        print(f"{fid}: not valid, skip persist")
        continue
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
    if not ok:
        print(f"{fid}: does not pass gate (IC={m['ic']:.4f} ICIR={m['icir']:.4f}), skip persist")
        continue
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
    # read-back verification
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    art = Path('factors') / payload['signal_artifact']
    arr2 = np.load(art)
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
