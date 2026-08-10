"""miner_2 2026-07-30 persist vol_price_corr_60 (batch-3 PASS)."""
import sys
sys.path.insert(0, 'scripts')
import json
import numpy as np
import pandas as pd
from factor_common import (load_prices, canonical_grid, factor_to_panel,
                           WATCHLIST, persist_factor)
import miner_2_20260730_screen_fast as scr

prices = load_prices(days=2000)
grid = canonical_grid(prices)


def f_vol_price_corr_60(df, s):
    r = df['close'].pct_change()
    v = df['volume']
    z = pd.concat([r.rename('r'), v.rename('v')], axis=1).replace([np.inf, -np.inf], np.nan)
    return z['r'].rolling(60).corr(z['v'])


panel = factor_to_panel(f_vol_price_corr_60, prices)
m = scr.evaluate_fast('vol_price_corr_60', panel)
assert m is not None
assert abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084, m
print("metrics:", json.dumps(m, indent=1, default=str))

path = persist_factor(
    factor_id='vol_price_corr_60',
    factor_name='Return-Volume Correlation (60d)',
    expression='corr(pct_change(close), volume, 60)',
    description=('Rolling 60-day Pearson correlation between daily returns and volume. '
                 'Negative IC: assets whose trading volume co-moves strongly with price '
                 'changes (high ret-vol correlation) tend to underperform over the 10-20d '
                 'forward horizon across the cross-asset universe; low/negative corr assets '
                 'outperform. Captures speculative churn / informed-flow congestion.'),
    dependencies=['close', 'volume'],
    parameters={'window': 60, 'min_periods': 30},
    expected_direction=-1,
    panel=panel,
    metrics=m,
    tags=['volume', 'liquidity', 'technical', 'cross_asset'],
    grid=grid,
    prices=prices,
    regime_notes=('2020-2026 multi-regime validation (COVID, rate shocks, crypto cycles). '
                  'Available on 9/15 assets (equity indices + crypto have volume; commodities/'
                  'yields lack volume series). Direction stable: negative IC at h>=5, '
                  'strengthens to h20.'),
)
print("persisted:", path)

# verify read-back
d = json.load(open(path))
v = d['validation']
ok = (d['factor_id'] == 'vol_price_corr_60'
      and v['status'] == 'EFFECTIVE'
      and abs(v['metrics']['ic']) >= 0.007
      and abs(v['metrics']['icir']) >= 0.084
      and d.get('signal_artifact'))
print("verify: id=%s status=%s ic=%.4f icir=%.4f art=%s rho=%.2f" % (
    d['factor_id'], v['status'], v['metrics']['ic'], v['metrics']['icir'],
    d.get('signal_artifact'), v['metrics'].get('max_abs_library_correlation')))
print("READBACK_OK" if ok else "READBACK_FAIL")
