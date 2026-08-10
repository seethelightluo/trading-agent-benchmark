"""miner_3 2026-07-30 persistence: wti_beta_60 and hs300_beta_60.

Both passed admission (|IC|>=0.007, |ICIR|>=0.084) with extended-library
max_abs_library_correlation < 0.5. Metrics recomputed here deterministically
and written together with the .npy signal artifact on the canonical grid.
"""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_common import (load_prices, factor_to_panel, validate_factor,
                           max_library_correlation, canonical_grid,
                           persist_factor, build_library_panels,
                           VAL_START, VAL_END, WATCHLIST)

prices = load_prices(days=2000)
print(f"loaded {len(prices)} assets")

def load_artifact_panels():
    out = {}
    grid = canonical_grid(prices)
    import glob, os
    for p in sorted(glob.glob('factors/*_signal.npy')):
        fid = os.path.basename(p).replace('_signal.npy', '')
        try:
            art = np.load(p, allow_pickle=False)
            if art.shape[0] == len(grid) and art.shape[1] == len(WATCHLIST):
                out[fid] = pd.DataFrame(art, index=grid, columns=WATCHLIST)
        except Exception as e:
            print('  artifact load fail', fid, e)
    return out

library_panels = build_library_panels(prices)
library_panels.update(load_artifact_panels())
print('extended library size for corr audit:', len(library_panels))

def beta_anchor(anchor_df):
    def fn(df, s):
        if anchor_df is None or len(anchor_df) < 70:
            return None
        r = df['close'].pct_change()
        ar = anchor_df['close'].pct_change()
        z = pd.concat([r.rename('r'), ar.rename('a')], axis=1, sort=True).dropna()
        b = z['r'].rolling(60).cov(z['a']) / z['a'].rolling(60).var()
        return b.reindex(z.index)
    return fn

specs = [
    dict(fid='wti_beta_60', anchor='WTI', direction=1,
         name='WTI commodity beta 60d',
         expr='BETA(pct_change(close,1), pct_change(WTI close,1), 60)',
         desc=('Rolling 60d ordinary least-squares beta of each asset\'s daily returns '
               'on WTI crude daily returns. Assets co-moving with the oil/commodity '
               'cycle (positive WTI beta) tend to keep outperforming over the next '
               '5-20d in this worldline: positive cross-sectional predictive power, '
               'monotone decay 1d 0.014 -> 20d 0.046. Low turnover (stable beta '
               'estimates) makes it a slow risk-on/commodity-cycle tilt.'),
         params=dict(anchor='WTI', window=60, min_obs=70),
         deps=['close', 'WTI close'],
         tags=['beta', 'commodity', 'cross-asset', 'risk-factor']),
    dict(fid='hs300_beta_60', anchor='000300.SH', direction=-1,
         name='CSI300 beta 60d',
         expr='BETA(pct_change(close,1), pct_change(000300.SH close,1), 60)',
         desc=('Rolling 60d beta of each asset\'s daily returns on CSI300 (000300.SH) '
               'returns. High China-exposure (high CSI300 beta) assets underperform '
               'over 10-20d horizons in this worldline: negative cross-sectional '
               'predictive power (IC -0.045, ICIR -0.125). Implemented as a defensive '
               'tilt against China-beta (long-only: underweight high-beta names).'),
         params=dict(anchor='000300.SH', window=60, min_obs=70),
         deps=['close', '000300.SH close'],
         tags=['beta', 'china', 'cross-asset', 'risk-factor']),
]

regime = ('Validated 2020-01-01..2026-07-15 on the 15-asset cross-asset universe '
          '(equity indices, commodities, crypto, yield series) across COVID crash '
          '2020, 2020-21 recovery bull, 2022 tightening bear, 2023-24 AI-led equity '
          'rally, 2024-26 crypto/commodity cycles. Daily cross-sectional Spearman IC '
          'vs 10d forward return, min 8 valid instruments per date.')

for spec in specs:
    fid = spec['fid']
    fn = beta_anchor(prices[spec['anchor']])
    panel = factor_to_panel(fn, prices)
    m = validate_factor(fid, panel, prices)
    if m is None:
        print(fid, ': insufficient data, skip'); continue
    rho, best = max_library_correlation(panel, library_panels)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = best
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
    print(f'{fid}: ic={m["ic"]:.4f} icir={m["icir"]:.4f} max_corr={rho:.3f} vs {best} PASS={ok}')
    if not ok or rho > 0.5:
        print(f'  -> NOT persisting ({fid})'); continue
    path, arr = persist_factor(
        factor_id=fid, factor_name=spec['name'], expression=spec['expr'],
        description=spec['desc'], dependencies=spec['deps'],
        parameters=spec['params'], expected_direction=spec['direction'],
        panel=panel, metrics=m, tags=spec['tags'], prices=prices,
        regime_notes=regime,
    )
    print('  persisted ->', path, 'artifact', arr.shape)
