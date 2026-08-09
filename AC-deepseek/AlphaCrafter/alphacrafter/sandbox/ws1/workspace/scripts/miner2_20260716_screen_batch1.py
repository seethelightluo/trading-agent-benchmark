"""Screening batch 1: price/volatility-based cross-asset factors (miner_2)."""
import sys
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from miner2_20260716_common import load_panel, evaluate_factor

panel = load_panel()
ret = panel.pct_change()
print(f'panel: {panel.shape[0]} weekday dates x {panel.shape[1]} instruments, through 2026-07-15')

MP = lambda w: max(10, int(0.7 * w))  # min_periods for rolling

factors = {}
factors['mom_20d'] = panel / panel.shift(20) - 1.0
factors['mom_60d'] = panel / panel.shift(60) - 1.0
factors['mom_10d'] = panel / panel.shift(10) - 1.0
factors['rev_5d'] = -(panel / panel.shift(5) - 1.0)
factors['lowvol_20d'] = -ret.rolling(20, min_periods=MP(20)).std()
factors['drawdown_120d'] = 1.0 - panel / panel.rolling(120, min_periods=MP(120)).max()
rng = panel.rolling(20, min_periods=MP(20)).max() - panel.rolling(20, min_periods=MP(20)).min()
factors['range_pos_20d'] = (panel - panel.rolling(20, min_periods=MP(20)).min()) / rng.replace(0, np.nan)
factors['neg_skew_60d'] = -ret.rolling(60, min_periods=MP(60)).skew()
wealth = (1 + ret).cumprod()
mdd = wealth / wealth.rolling(60, min_periods=MP(60)).max() - 1.0
factors['shallow_dd_60d'] = -mdd.rolling(60, min_periods=MP(60)).min()
factors['rel_strength_20d'] = (panel / panel.shift(20) - 1.0).subtract(
    (panel / panel.shift(20) - 1.0).mean(axis=1), axis=0)
mom5 = panel / panel.shift(5) - 1.0
mom20 = panel / panel.shift(20) - 1.0
factors['accel_5_20'] = mom5 - mom20

rows = []
for name, f in factors.items():
    res, to, tc, cov = evaluate_factor(f, panel, horizons=(5, 10, 20, 40))
    def g(h, k):
        return res[h][k] if res[h] else np.nan
    rows.append({
        'factor': name,
        'ic5': g(5, 'mean_ic'), 'ic10': g(10, 'mean_ic'), 'ic20': g(20, 'mean_ic'),
        'ic40': g(40, 'mean_ic'), 'icir20': g(20, 'icir'), 'hit20': g(20, 'hit_ratio'),
        't20': g(20, 't_stat'), 'n20': g(20, 'n_dates'), 'turnover': to,
        'coverage': round(cov['coverage'], 4),
    })

tbl = pd.DataFrame(rows).sort_values('ic20', key=lambda s: s.abs(), ascending=False)
pd.set_option('display.width', 220)
print(tbl.to_string(index=False))
