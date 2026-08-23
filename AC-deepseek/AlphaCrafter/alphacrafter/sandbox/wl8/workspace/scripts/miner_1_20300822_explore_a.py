"""miner_1 2030-08-22 factor exploration batch A:
C1 stress_mom_5x63: 5d momentum gated by 63d trend sign, vol-normalized (crisis/momentum-shock edge).
C2 vix_chg_5x10: 5d change in log-VIX (fear delta) - direction from sign of IC.
Validation through visible_through (2030-08-21) only.
"""
import sys, json, math
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from miner_1_common import (load_panel, load_macro_panel, forward_returns, spearman_ic_series,
                            ic_metrics, regime_slices, decay_by_horizon, zlib_b64_panel,
                            max_library_corr, IC_THRESHOLD, ICIR_THRESHOLD, ADMISSION_HORIZON)

panel, vpanel = load_panel()
print('panel shape:', panel.shape, 'dates:', panel.index.min().date(), '->', panel.index.max().date())
fwd = forward_returns(panel, ADMISSION_HORIZON)
macro = {m: load_macro_panel(m) for m in ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']}

# Candidate 1: stress_mom_5x63
r1 = panel.pct_change()
vol5 = r1.rolling(5).std()
mom5 = panel / panel.shift(5) - 1.0
mom63 = panel / panel.shift(63) - 1.0
f1 = np.sign(mom63) * mom5 / (vol5 + 1e-9)
ics1 = spearman_ic_series(f1, fwd)
m1 = ic_metrics(ics1)
print('C1 stress_mom_5x63 ic:', round(m1['ic'], 4), 'icir:', round(m1['icir'], 4),
      'n:', m1['n_ic_dates'], 'hit:', round(m1['hit'], 3))
print('   regimes:', regime_slices(ics1))
print('   decay:', decay_by_horizon(panel, f1))
cov1 = float(f1.notna().sum().sum() / (panel.shape[0] * panel.shape[1]))
print('   coverage_asset_days:', round(cov1, 4))
print('   turnover_rank_chg:', round(float(f1.rank(axis=1).diff().abs().mean()), 4))
corr1 = max_library_corr(f1, panel)
print('   max_lib_corr:', corr1[0], {k: round(v, 3) for k, v in corr1[1].items() if not np.isnan(v)})

# Candidate 2: VIX regime factors (macro-conditional)
lv = np.log(macro['VIX'])
d_lv = lv.diff()
for name, f2 in [('dvix_5', d_lv.rolling(5).sum()),
                 ('dvix_5pos', np.clip(d_lv.rolling(5).sum(), 0, None)),
                 ('dvix_5neg', np.clip(d_lv.rolling(5).sum(), None, 0))]:
    fdf = pd.DataFrame({s: f2 for s in panel.columns}, index=panel.index)
    ics = spearman_ic_series(fdf, fwd)
    mm = ic_metrics(ics)
    print(f'C2 {name}: ic {round(mm["ic"],4)} icir {round(mm["icir"],4)} n {mm["n_ic_dates"]} hit {round(mm["hit"],3)}')
    print('   regimes:', regime_slices(ics))