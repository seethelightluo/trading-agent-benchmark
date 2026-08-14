"""miner_1 pipeline sanity test: recompute vol_adj_mom_accel_20x60 and run IC."""
import sys
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
import miner_1_factorlib as fl

closes = fl.load_closes()
rets = closes.pct_change()

fast, slow, volw = 20, 60, 20
mom_fast = closes / closes.shift(fast) - 1.0
mom_slow = closes / closes.shift(slow) - 1.0
vol = rets.rolling(volw).std()
factor = (mom_fast - mom_slow) / vol

lib = fl.load_library_signals()
res = fl.evaluate(factor, closes, lib_signals=lib)
print('pipeline test: vol_adj_mom_accel_20x60 recomputed')
for h, m in res['metrics_by_horizon'].items():
    print(f"  h={h:>2s}  IC={m['ic']:+.5f}  ICIR={m['icir']:+.5f}  hit={m['ic_hit_ratio']:.3f}  n={m['n_ic_dates']}")
print('  coverage:', res['coverage'])
print('  turnover_10d_rank:', res['turnover_10d_rank'])
print('  max_abs_library_correlation:', res.get('max_abs_library_correlation'),
      'vs', res.get('max_corr_factor'))
print('  n_dates x n_assets:', factor.shape)
