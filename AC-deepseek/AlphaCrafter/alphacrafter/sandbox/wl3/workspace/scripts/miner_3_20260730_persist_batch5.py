"""miner_3 2026-07-30: persist round-4 passing factors (batch5 winners).

Winners:
  - idio_vol_ratio_60 : 1 - R2 of 60d regression on equal-weight basket
    IC=-0.0485 ICIR=-0.1252 rho=0.164 -> direction -1 (high idiosyncratic
    share underperforms: broad participation rewarded).
  - copper_beta_60 : rolling 60d beta to COPPER returns
    IC=+0.0650 ICIR=+0.1794 rho=0.058 -> direction +1 (cyclical beta premium).

Both artifacts saved to canonical grid (2388 x 15) as required by the gate.
"""
import sys, json
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from factor_common import (load_prices, factor_to_panel, validate_factor,
                           max_library_correlation, canonical_grid,
                           VAL_START, VAL_END, WATCHLIST, persist_factor,
                           load_artifact_matrix, Path)

prices = load_prices(days=2100)
grid = canonical_grid(prices)
print(f"loaded {len(prices)} assets; grid {grid.min().date()}..{grid.max().date()} n={len(grid)}")

ret_panel = pd.DataFrame({s: d['close'].pct_change() for s, d in prices.items()})
basket = ret_panel.mean(axis=1)
copper = prices.get('COPPER')


def load_effective_artifact_panels():
    out = {}
    for jp in sorted(Path('factors').glob('*.json')):
        if jp.name == 'factor_ensemble.json':
            continue
        payload = json.loads(jp.read_text(encoding='utf-8'))
        if payload.get('validation', {}).get('status') != 'EFFECTIVE':
            continue
        art = load_artifact_matrix(str(jp))
        if art is None or art.shape[0] != len(grid):
            continue
        out[payload['factor_id']] = pd.DataFrame(art, index=grid, columns=WATCHLIST)
    return out


def idio_vol_ratio_60(df, s):
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), basket.rename('b')], axis=1).dropna()
    var_b = z['b'].rolling(60, min_periods=30).var()
    cov = z['r'].rolling(60, min_periods=30).cov(z['b'])
    var_r = z['r'].rolling(60, min_periods=30).var()
    r2 = (cov ** 2 / (var_b * var_r)).replace([np.inf, -np.inf], np.nan)
    return (1.0 - r2).reindex(z.index)


def copper_beta_60(df, s):
    if copper is None:
        return None
    r = df['close'].pct_change()
    rc = copper['close'].pct_change()
    z = pd.concat([r.rename('r'), rc.rename('c')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['c']) / z['c'].rolling(60).var().replace(0, np.nan)
    return b.reindex(z.index)


library = load_effective_artifact_panels()
print('library for corr audit:', sorted(library.keys()))

cands = [
    dict(fid='idio_vol_ratio_60',
         name='Idiosyncratic Volatility Share 60d',
         expr='1 - R2(ret, basket, 60)  [rolling 60d regression on equal-weight 15-asset basket]',
         desc='Share of asset return variance NOT explained by the equal-weight cross-asset '
              'basket over 60d. Low idio share = high participation in global risk moves; '
              'high idio share = idiosyncratic/decoupled asset. Negative IC: high idio share '
              'underperforms over 10d (convergence/breadth premium).',
         deps=['close'], params={'window': 60, 'min_periods': 30, 'basket': 'equal-weight 15 assets'},
         direction=-1, fn=idio_vol_ratio_60, tags=['idiosyncratic-risk', 'breadth', 'cross-asset'],
         notes='Warm-up 2020-01..2026-07; regimes: COVID crash, 2022 tightening, 2023-25 risk-on, '
               '2025-26 crypto/commodity cycles. Direction -1 (IC<0).'),
    dict(fid='copper_beta_60',
         name='Copper Beta 60d',
         expr='rolling 60d beta of asset returns to COPPER returns',
         desc='Sensitivity to the cyclical industrial-commodity anchor (copper = global growth '
              'barometer). Positive IC: assets with high copper beta earn a cyclical-growth '
              'premium over 10d. Distinct anchor vs SPX/HSI/DXY/VIX conditional betas in library.',
         deps=['close'], params={'window': 60, 'anchor': 'COPPER'},
         direction=1, fn=copper_beta_60, tags=['beta', 'cyclical', 'commodity'],
         notes='Warm-up 2020-01..2026-07; copper led 2021 reflation, 2022-23 China slowdown, '
               '2024-26 supply/energy cycles. Direction +1 (IC>0).'),
]

for c in cands:
    panel = factor_to_panel(c['fn'], prices)
    m = validate_factor(c['fid'], panel, prices)
    rho, best = max_library_correlation(panel, library)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = best
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084 and rho < 0.5
    print(f"{c['fid']}: ic={m['ic']:+.4f} icir={m['icir']:+.4f} rho={rho:.3f} vs {best} -> {'PERSIST' if ok else 'SKIP'}")
    if not ok:
        continue
    path, arr = persist_factor(
        factor_id=c['fid'], factor_name=c['name'], expression=c['expr'],
        description=c['desc'], dependencies=c['deps'], parameters=c['params'],
        expected_direction=c['direction'], panel=panel, metrics=m, tags=c['tags'],
        grid=grid, prices=None, version='1.0.0', status='EFFECTIVE',
        regime_notes=c['notes'])
    print(f'  wrote {path} artifact {arr.shape}')
