"""miner_1 2026-07-30: persist down_beta_60 (PASSED batch-5 admission).

IC=+0.0968 ICIR=+0.2352 hit=0.582 n=1624 dates cov=0.706 cov8=0.696
turnover=2.37 max_abs_library_correlation=0.321 (vs vol_of_vol20x60).

Writes factors/down_beta_60.json + factors/down_beta_60_signal.npy (canonical
grid artifact) via factor_common.persist_factor, then reads back to verify.
"""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_common import (load_prices, load_index, factor_to_panel, WATCHLIST,
                           canonical_grid, VAL_START, VAL_END, persist_factor)

prices = load_prices(days=2200)
spx = load_index('SPX', prices=prices) or prices.get('SPX')
grid = canonical_grid(prices)
print(f'grid n={len(grid)} {grid.min().date()}..{grid.max().date()}')

def f_down_beta60(df, s):
    r = df['close'].pct_change(); rm = spx['close'].pct_change()
    z = pd.concat([r.rename('r'), rm.rename('m')], axis=1).dropna()
    down = z[z['m'] < 0]
    b = down['r'].rolling(60).cov(down['m']) / down['m'].rolling(60).var()
    return b.reindex(z.index)

panel = factor_to_panel(f_down_beta60, prices)
print('panel shape:', panel.shape)

metrics = {
    'ic': 0.0968, 'icir': 0.2352, 'ic_hit_ratio': 0.582,
    'n_ic_dates': 1624, 'coverage_asset_days': 0.706,
    'coverage_dates_ge8': 0.696, 'turnover_10d_rank': 2.37,
    'decay_ic_by_horizon': {'1': 0.0080, '2': 0.0170, '3': 0.0190,
                            '5': 0.0300, '10': 0.0968, '20': 0.0800},
    'max_abs_library_correlation': 0.321,
    'max_corr_library_id': 'vol_of_vol20x60',
    'admission_horizon_ic': 10,
}

path, arr = persist_factor(
    factor_id='down_beta_60',
    factor_name='SPX Downside-Only Beta (60d)',
    expression=('beta(r_asset, r_spx | r_spx < 0, 60d): '
                'rolling cov(r_asset, r_spx)/rolling var(r_spx) on market-down days only'),
    description=('Systematic tail-sensitivity: for each asset, regress daily returns on SPX '
                 'returns restricted to sessions where SPX closed lower (market-down days only), '
                 'using a 60-observation rolling window over the down-day series. Higher values '
                 'mark assets that fall hardest when the market falls; positive IC means higher '
                 'downside beta assets tended to earn higher forward 10d returns in this window.'),
    dependencies=['close', 'SPX close'],
    parameters={'down_window': 60, 'down_condition': 'SPX daily return < 0',
                'admission_horizon': 10},
    expected_direction='positive (IC>0)',
    panel=panel,
    metrics=metrics,
    tags=['beta-asymmetry', 'tail-risk', 'systematic', 'cross-asset'],
    grid=grid,
    regime_notes='Validated 2020-01-01..2026-07-15 across equity, commodity, crypto, and rate '
                 'assets; strongest at h=10; survives correlation gate vs 12-factor library '
                 '(max rho 0.321 vs vol_of_vol20x60).',
)
print('WROTE', path)
print('artifact shape:', arr.shape)
