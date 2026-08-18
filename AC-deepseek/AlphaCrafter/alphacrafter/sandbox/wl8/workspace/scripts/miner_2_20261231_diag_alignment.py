"""miner_2 diagnostic 2026-12-31: understand date alignment & coverage for honest IC reporting."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from miner_3_20261203_common import WATCH, load_prices

ASOF = '2026-12-30'
px = load_prices(ASOF)

print('px shape:', px.shape, 'index range:', px.index.min().date(), '->', px.index.max().date())
print('total dates:', len(px))
wd = px.index.dayofweek < 5
print('weekday dates:', int(wd.sum()), 'weekend dates:', int((~wd).sum()))

# per asset valid days
valid = px.notna()
per_asset = valid.sum()
print('\nper-asset valid days:')
for s in WATCH:
    print(f'  {s:10s} {per_asset[s]:5d}  first={px.index[valid[s]].min().date()} last={px.index[valid[s]].max().date()}')

# dates with >=8 valid (any), on weekdays
n_all = ((valid.sum(axis=1) >= 8)).sum()
n_wd = ((valid.sum(axis=1) >= 8) & wd).sum()
print(f'\ndates with >=8 valid assets: all={n_all}, weekday={n_wd}')

# factor validity for mom10: needs 15d history; accel needs 65d
import numpy as np
for name, lookback in [('mom10_skip5', 15), ('accel_20v60', 65), ('trend_quality_20', 80)]:
    f = px / px.shift(lookback) - 1.0
    fv = f.notna()
    cov = fv.mean().mean()
    icd_all = ((fv.sum(axis=1) >= 8)).sum()
    icd_wd = (((fv.sum(axis=1) >= 8)) & wd).sum()
    print(f'{name:18s} cov={cov:.3f} dates>=8valid: all={icd_all:5d} weekday={icd_wd:5d}')

# check what weekdays look like: how many have >=8 assets valid (raw close)
# and how many weekday-IC-date losses come from weekend contamination of fwd (10d fwd spans weekends too)
fwd = px.shift(-10) / px - 1.0
fwdv = fwd.notna()
icd_fwd_wd = (((fwdv.sum(axis=1) >= 8)) & wd).sum()
print(f'\nfwd-valid (>=8 assets, weekday dates): {icd_fwd_wd}')

# empty-market check: count dates where <8 valid raw closes on a weekday
lose = ((valid.sum(axis=1) < 8) & wd).sum()
print(f'weekday dates with <8 assets having ANY close: {lose}')