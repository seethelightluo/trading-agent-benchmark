"""miner_2 2026-07-30 persist batch-2 passers: eurusd_beta_cond_60x20, spx_beta_60, btc_beta_60.

Computes max_abs_library_correlation against BOTH the 4 legacy panels and the
5 persisted effective artifacts, then persists via persist_factor (JSON + .npy).
"""
import sys
sys.path.insert(0, 'scripts')
import json
from pathlib import Path
import numpy as np
import pandas as pd
from factor_common import (load_prices, load_index, WATCHLIST, canonical_grid,
                           factor_to_panel, persist_factor, build_library_panels)
import miner_2_20260730_screen_fast as scr
import miner_2_20260730_screen_batch2 as b2

prices = load_prices(days=2000)
grid = canonical_grid(prices)

# full library: 4 legacy panels + 5 persisted artifact matrices
lib_panels = build_library_panels(prices)
lib_mat = {}
for fid, lp in lib_panels.items():
    lib_mat[fid] = lp.reindex(grid)[WATCHLIST].values.astype(float)
for f in sorted(Path('factors').glob('*_signal.npy')):
    fid = f.name.replace('_signal.npy', '')
    arr = np.load(f)
    if arr.shape == (len(grid), 15):
        lib_mat[fid] = arr
print(f"full library for rho audit ({len(lib_mat)}): {sorted(lib_mat)}")


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


FACTORS = [
    {
        'fid': 'eurusd_beta_cond_60x20',
        'name': 'EURUSD-beta conditional 60x20',
        'expr': 'BETA(ret, EURUSD_ret, 60) * (EURUSD / EURUSD.shift(20) - 1)',
        'desc': 'Rolling 60d beta of asset returns to EURUSD returns, scaled by EURUSD 20d '
                'momentum (EUR carry / risk channel). Positive IC -> direction +1 on raw value.',
        'deps': ['close', 'EURUSD(obs)'],
        'params': {'beta_window': 60, 'move_window': 20},
        'direction': 1,
        'tags': ['cross_asset', 'fx_beta', 'conditional'],
        'regime': '2020-2026 multi-regime; IC strengthens with horizon (h10 0.034, h20 0.042).',
    },
    {
        'fid': 'spx_beta_60',
        'name': 'SPX rolling beta 60d',
        'expr': 'BETA(ret, SPX_ret, 60)',
        'desc': 'Rolling 60d beta of each asset to SPX returns (systematic risk exposure). '
                'High-beta assets outperform over 10-20d (risk-on tilt in this worldline). '
                'Positive IC -> direction +1.',
        'deps': ['close', 'SPX(obs)'],
        'params': {'beta_window': 60},
        'direction': 1,
        'tags': ['cross_asset', 'beta', 'risk'],
        'regime': '2020-2026 multi-regime; strongest horizon 10-20d (h20 IC 0.12).',
    },
    {
        'fid': 'btc_beta_60',
        'name': 'BTC rolling beta 60d',
        'expr': 'BETA(ret, BTC_ret, 60)',
        'desc': 'Rolling 60d beta of each asset to BTC returns (crypto/risk-on factor loading). '
                'Assets with high BTC sensitivity outperform over 10d. Positive IC -> direction +1.',
        'deps': ['close', 'BTC(obs)'],
        'params': {'beta_window': 60},
        'direction': 1,
        'tags': ['cross_asset', 'beta', 'crypto'],
        'regime': '2020-2026 multi-regime; coverage lower (crypto listing gaps).',
    },
]

for f in FACTORS:
    fid = f['fid']
    fn = dict(b2.CANDIDATES)[fid]
    panel = factor_to_panel(fn, prices)
    m = scr.evaluate_fast(fid, panel)
    assert m is not None and abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084, f"{fid}: {m}"
    fac = panel.reindex(grid)[WATCHLIST].values.astype(float)
    rho, rho_id = max_rho(fac)
    metrics = {k: m[k] for k in ['ic', 'icir', 'ic_hit_ratio', 'n_ic_dates',
                                 'coverage_asset_days', 'coverage_dates_ge8',
                                 'turnover_10d_rank', 'decay_ic_by_horizon']}
    metrics['max_abs_library_correlation'] = rho
    metrics['max_corr_library_id'] = rho_id
    print(f"{fid}: full-lib rho={rho:.3f} vs {rho_id} (gate threshold 0.5)")
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
          f"rho_lib={rho:.2f}")
