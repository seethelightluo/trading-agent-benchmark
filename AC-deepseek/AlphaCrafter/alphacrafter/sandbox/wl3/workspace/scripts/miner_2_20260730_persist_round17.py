"""miner_2 2026-07-30 persist round-17 winners: gain_loss_asym_20 + copper_gold_beta_20.

Both passed the benchmark-wide admission gate in miner_2_20260730_results_round17.json:
  gain_loss_asym_20 : |IC|=0.0431>=0.007, |ICIR|=0.1287>=0.084, rho_lib=0.130<0.5
  copper_gold_beta_20: |IC|=0.0302>=0.007, |ICIR|=0.0844>=0.084, rho_lib=0.210<0.5
Recomputes panels + validation, persists JSON + npy signal artifact, verifies reload.
"""
import sys, json, time
sys.path.insert(0, 'scripts')
from pathlib import Path
import numpy as np
import pandas as pd
from factor_common import (load_prices, factor_to_panel, persist_factor,
                           canonical_grid, signal_matrix, validate_factor,
                           WATCHLIST)
import miner_2_20260730_screen_round17_novel as scr

t0 = time.time()
prices = load_prices(days=2000)
grid = canonical_grid(prices)
print(f'prices={len(prices)} grid={len(grid)} {grid.min().date()}..{grid.max().date()}', flush=True)

# Recompute library corr exactly as the round-17 screen did (effective factors only).
lib = {}
for p in sorted(Path('factors').glob('*.json')):
    if p.name.endswith('.bak') or 'deprecated' in p.name:
        continue
    try:
        payload = json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        continue
    if payload.get('validation', {}).get('status') != 'EFFECTIVE':
        continue
    art = payload.get('signal_artifact')
    art_path = p.parent / str(art) if art else None
    if art_path is not None and art_path.exists():
        lib[payload['factor_id']] = np.load(art_path, allow_pickle=False)
print(f'library factors: {len(lib)} {sorted(lib.keys())}', flush=True)

def lib_max_corr(panel):
    arr = signal_matrix(panel, grid)
    best, best_id = 0.0, None
    for fid, larr in lib.items():
        corrs = []
        for i in range(arr.shape[0]):
            x, y = arr[i], larr[i]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 8:
                xr = pd.Series(x[m]).rank().values
                yr = pd.Series(y[m]).rank().values
                c = np.corrcoef(xr, yr)[0, 1]
                if np.isfinite(c):
                    corrs.append(c)
        if corrs:
            r = float(np.mean(corrs))
            if abs(r) > best:
                best, best_id = abs(r), fid
    return best, best_id

FACTORS = [
    {
        'fid': 'gain_loss_asym_20',
        'name': 'Gain/Loss asymmetry 20d',
        'expr': 'MEAN(MAX(pct_change(close,1),0),20) / (MEAN(-MIN(pct_change(close,1),0),20) + eps)',
        'desc': 'Ratio of the 20-day mean positive daily return to the 20-day mean '
                'absolute negative daily return. Assets with high gain/loss asymmetry '
                '(persistent upside capture relative to downside) outperform over 10-20d '
                'horizons in this worldline: positive cross-sectional predictive power '
                '(IC +0.043, ICIR +0.129). Long-only: overweight high-asymmetry assets.',
        'deps': ['close'],
        'params': {'window': 20, 'eps': 1e-9},
        'direction': 1,
        'tags': ['asymmetry', 'momentum', 'cross-asset'],
        'regime': 'Validated 2020-01-01..2026-07-15 on the 15-asset cross-asset universe '
                  'across COVID crash 2020, 2020-21 recovery bull, 2022 tightening bear, '
                  '2023-24 AI-led equity rally, 2024-26 crypto/commodity cycles. IC grows '
                  'monotonically with horizon (1d +0.002 -> 20d +0.058); strongest at 10-20d.',
    },
    {
        'fid': 'copper_gold_beta_20',
        'name': 'Copper-Gold rotation beta 20d',
        'expr': 'BETA(pct_change(close,1), pct_change(COPPER,1) - pct_change(XAU,1), 20)',
        'desc': 'Rolling 20d beta of each asset on the copper-minus-gold return spread '
                '(cyclical vs safe-haven rotation). Assets positively exposed to the '
                'copper/gold rotation spread outperform over 10-20d: positive '
                'cross-sectional predictive power (IC +0.030, ICIR +0.084). High coverage '
                '(~94% asset-days, ~99% dates with >=8 valid instruments).',
        'deps': ['close', 'COPPER close', 'XAU close'],
        'params': {'window': 20, 'spread': 'COPPER_ret - XAU_ret', 'min_obs': 30},
        'direction': 1,
        'tags': ['beta', 'cross-asset', 'rotation', 'commodity'],
        'regime': 'Validated 2020-01-01..2026-07-15 on the 15-asset cross-asset universe '
                  'across COVID crash 2020, 2020-21 recovery bull, 2022 tightening bear, '
                  '2023-24 AI-led equity rally, 2024-26 crypto/commodity cycles. IC rises '
                  'with horizon (1d +0.010 -> 20d +0.044); robust at 10-20d.',
    },
]

ok_all = True
for f in FACTORS:
    fid = f['fid']
    fn = dict(scr.CANDIDATES)[fid]
    panel = factor_to_panel(fn, prices)
    m = validate_factor(fid, panel, prices)
    if m is None:
        print(f'{fid}: insufficient data -> None', flush=True)
        ok_all = False
        continue
    rho, rho_id = lib_max_corr(panel)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = rho_id
    ic_ok = abs(m['ic']) >= 0.007
    icir_ok = abs(m['icir']) >= 0.084
    corr_ok = rho < 0.5
    ok = ic_ok and icir_ok and corr_ok
    print(f'{fid}: panel={panel.shape} IC={m["ic"]:.4f} ICIR={m["icir"]:.4f} '
          f'rho_lib={rho:.3f}({rho_id}) pass={ok}', flush=True)
    if not ok:
        ok_all = False
        continue

    metrics = {k: m[k] for k in ['ic', 'icir', 'ic_hit_ratio', 'n_ic_dates',
                                 'coverage_asset_days', 'coverage_dates_ge8',
                                 'turnover_10d_rank', 'decay_ic_by_horizon',
                                 'max_abs_library_correlation', 'max_corr_library_id']}
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
    print(f'PERSISTED {fid} -> {path}', flush=True)

    # ---- read back and verify ----
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    arr2 = np.load(Path('factors') / payload['signal_artifact'], allow_pickle=False)
    checks = {
        'json_valid': True,
        'factor_id_ok': payload['factor_id'] == fid,
        'status_ok': payload['validation']['status'] == 'EFFECTIVE',
        'ic_ok': abs(payload['validation']['metrics']['ic']) >= 0.007,
        'icir_ok': abs(payload['validation']['metrics']['icir']) >= 0.084,
        'artifact_shape_ok': arr2.shape == (len(grid), len(WATCHLIST)),
        'artifact_matches': np.allclose(arr2, arr, equal_nan=True),
        'corr_reported': 'max_abs_library_correlation' in payload['validation']['metrics'],
    }
    print(f'VERIFY {fid}: {checks}', flush=True)
    assert all(checks.values()), f'verification failed for {fid}'
    print(f'OK {fid} persisted+reloadable IC={metrics["ic"]:.4f} ICIR={metrics["icir"]:.4f} '
          f'rho_lib={rho:.3f}', flush=True)

print(f'elapsed={time.time()-t0:.1f}s all_ok={ok_all}', flush=True)
