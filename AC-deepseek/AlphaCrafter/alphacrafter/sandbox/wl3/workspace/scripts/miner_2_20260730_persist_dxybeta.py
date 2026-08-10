"""miner_2 2026-07-30 persist dxy_beta_cond_60x20 (passed gate)."""
import sys
sys.path.insert(0, 'scripts')
import json
from pathlib import Path
import numpy as np
from factor_common import (load_prices, load_index, factor_to_panel, persist_factor,
                           canonical_grid)
import miner_2_20260730_screen_fast as scr

prices = load_prices(days=2000)
grid = canonical_grid(prices)

fid = 'dxy_beta_cond_60x20'
fn = dict(scr.CANDIDATES)[fid]
panel = factor_to_panel(fn, prices)
m = scr.evaluate_fast(fid, panel)
assert m is not None and abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084, m
metrics = {k: m[k] for k in ['ic', 'icir', 'ic_hit_ratio', 'n_ic_dates',
                             'coverage_asset_days', 'coverage_dates_ge8',
                             'turnover_10d_rank', 'max_abs_library_correlation',
                             'decay_ic_by_horizon']}
path, arr = persist_factor(
    factor_id=fid,
    factor_name='DXY beta conditional 60x20',
    expression='beta(asset_ret, DXY_ret, 60) * (DXY/ DXY.shift(20) - 1)',
    description='Asset sensitivity to USD (60d rolling beta to DXY) interacted with the '
                '20d DXY move. Assets with high positive DXY beta tend to outperform when '
                'the dollar strengthens over the prior month; the conditional interaction '
                'captures cross-asset dollar-regime tilts. Positive IC -> direction +1.',
    dependencies=['close', 'DXY_close'],
    parameters={'beta_window': 60, 'dxy_lookback': 20},
    expected_direction=1,
    panel=panel,
    metrics=metrics,
    tags=['macro', 'cross_asset', 'currency_beta', 'conditional'],
    grid=grid,
    prices=prices,
    version='1.0.0',
    status='EFFECTIVE',
    regime_notes='2020-2026 includes USD up/down regimes (COVID flight-to-dollar, 2022 '
                 'Fed tightening, 2023-2025 dollar swings); signal strongest at 5-20d horizon.',
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
assert all(checks.values())
print(f"OK {fid} persisted and reloadable, IC={metrics['ic']:.4f} ICIR={metrics['icir']:.4f} "
      f"rho_lib={metrics['max_abs_library_correlation']:.2f}")
