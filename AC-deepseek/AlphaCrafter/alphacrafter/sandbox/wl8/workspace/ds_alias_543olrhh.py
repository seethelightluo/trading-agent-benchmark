import pandas as pd, json, numpy as np, math, zlib, base64, os, sys, glob
sys.path.insert(0, 'scripts')
from miner_1_common import (load_panel, load_macro_panel, forward_returns, spearman_ic_series,
                            ic_metrics, regime_slices, decay_by_horizon, zlib_b64_panel,
                            max_library_corr, IC_THRESHOLD, ICIR_THRESHOLD, ADMISSION_HORIZON,
                            WATCHLIST, visible_through)
panel, vpanel = load_panel()
print('panel shape:', panel.shape, 'dates:', panel.index.min().date(), '->', panel.index.max().date())
fwd = forward_returns(panel, ADMISSION_HORIZON)
macro = {}
for m in ['DXY','USDCNY','USDJPY','EURUSD','VIX']:
    macro[m] = load_macro_panel(m)

# ---------------- Candidate 1: stress_mom_5x63 ----------------
# 5d momentum gated by 63d trend sign, vol-normalized
r1 = panel.pct_change()
vol5 = r1.rolling(5).std()
mom5 = panel.panel? None
mom5 = panel / panel.shift(5) - 1.0
mom63 = panel / panel.shift(63) - 1.0
f1 = np.sign(mom63) * mom5 / (vol5 + 1e-9)
ics1 = spearman_ic_series(f1, fwd)
m1 = ic_metrics(ics1)
print('C1 stress_mom_5x63:', {k: (round(v,4) if isinstance(v,float) else v) for k,v in m1.items()})
print('   regimes:', regime_slices(ics1))
print('   decay:', decay_by_horizon(panel, f1))
cov1 = float((f1.loc[f1.index.intersection(panel.index)].notna().sum().sum())/(panel.shape[0]*panel.shape[1]))
print('   coverage_asset_days:', round(cov1,4))
print('   turnover_10d:', round(float(f1.rank(axis=1).diff().abs().mean()),4))
corr1 = max_library_corr(f1, panel)
print('   max_lib_corr:', corr1[0], {k: round(v,3) for k,v in corr1[1].items() if not np.isnan(v)})