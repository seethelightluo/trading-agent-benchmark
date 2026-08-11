"""Trader dry-run for 2026-09-10: replicate strategy_hook but capture the
rebalance proposal instead of mutating the live account."""
import json
import math
import sys
sys.path.insert(0, '.')

import strategy as S

captured = {}
def fake_rebalance(weights, forecast_returns=None, factor_ids=None, horizon_days=10, **kw):
    captured['weights'] = weights
    captured['forecast'] = forecast_returns
    captured['factor_ids'] = factor_ids
    captured['horizon'] = horizon_days

S.rebalance_to_weights = fake_rebalance

print('is_block_start:', S.is_block_start())
print('ensemble size:', len(S.load_ensemble()))
S.strategy_hook()

if not captured:
    print('NO PROPOSAL PRODUCED')
    sys.exit(0)

w = captured['weights']
assets = sorted(w.keys())
print('n_assets:', len(assets))
print('sum weights:', round(sum(w.values()), 10))
print('min weight:', round(min(w.values()), 10), 'max:', round(max(w.values()), 10))
print('any negative:', any(v < 0 for v in w.values()))
print('any NaN:', any(math.isnan(v) for v in w.values()))
print()
print('target weights:')
for a in sorted(w, key=lambda x: -w[x]):
    print(f'  {a:12s} {w[a]*100:6.2f}%')
print()
print('forecast (bps):')
for a in sorted(captured['forecast'], key=lambda x: -captured['forecast'][x]):
    print(f'  {a:12s} {captured["forecast"][a]*10000:8.2f} bp')
print('factor_ids:', captured['factor_ids'])
print('horizon:', captured['horizon'])
