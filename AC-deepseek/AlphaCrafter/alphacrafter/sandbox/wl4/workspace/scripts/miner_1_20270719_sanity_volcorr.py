"""Sanity check: reproduce vol_price_corr_20 IC/ICIR on its own validation
window to confirm the harness matches the persisted admission metrics."""
import sys
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from factor_lib import (load_asset_panel, evaluate_factor, gates_pass,
                        load_library_panels, max_abs_library_correlation)

panel = load_asset_panel()
ret = panel.pct_change()

# vol_price_corr_20: rolling corr(return, volume, 20)
vol = None
from alphacrafter.sim.utils import get_stock_daily_data
vols = {}
for s in panel.columns:
    df = get_stock_daily_data(symbol=s, days=3000)
    df['date'] = pd.to_datetime(df['date'])
    vols[s] = df.set_index('date')['volume']
volpanel = pd.DataFrame(vols).sort_index()
factor = ret.rolling(20, min_periods=10).corr(volpanel)

m = evaluate_factor(factor, panel, h=10, label='vol_price_corr_20',
                    valid_from='2020-01-01', valid_to='2026-07-15')
for k, v in m.items():
    if k != 'decay_ic_by_horizon':
        print(f'{k}: {v}')
print('decay:', m['decay_ic_by_horizon'])
ok, ic, icir = gates_pass(m)
print(f'GATES: |IC|={ic:.4f} (>=0.0070) |ICIR|={icir:.4f} (>=0.0840) -> PASS={ok}')
