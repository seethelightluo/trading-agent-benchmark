# -*- coding: utf-8 -*-
"""miner_1 2030-10-31: deep-dive validation of cny_beta_60 before persistence."""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from miner_1_20301114_common import (
    WATCH, VISIBLE_THROUGH, CURRENT_DATE, ohlcv_panels, macro_panel,
    rank_ic_series, summarize_ic, decay_analysis, turnover_10d, coverage_stats,
    regime_split, roll_beta, library_correlation,
)

C = ohlcv_panels()['close']
R = C.pct_change()
USDCNY = macro_panel('USDCNY')
RCNY = USDCNY.pct_change()
fwd10 = C.shift(-10) / C - 1.0

beta = roll_beta(R, RCNY, 60, 30)

# 1) who has high/low CNY beta currently
last = beta.index[-1]
print('cny_beta_60 as of %s:' % last.date())
print(beta.iloc[-1].sort_values(ascending=False).round(4).to_string())

# 2) yearly IC breakdown
ic_s = rank_ic_series(beta, fwd10)
print('\nYearly IC (10d horizon):')
for yr in range(2020, 2031):
    sub = ic_s[(ic_s.index >= pd.Timestamp(f'{yr}-01-01')) & (ic_s.index <= pd.Timestamp(f'{yr}-12-31'))]
    if len(sub):
        print(f'  {yr}: IC={sub.mean():+.4f} ICIR={sub.mean()/sub.std(ddof=1)*np.sqrt(len(sub)):+.2f} n={len(sub)}')

# 3) rolling 180d mean IC (non-overlapping monthly)
roll = ic_s.rolling(180).mean()
print('\nRolling 180d IC (last 12 monthly obs):')
print(roll.dropna().iloc[-12:].round(4).to_string())

# 4) summary + library correlation detail
m = summarize_ic(ic_s, 'cny_beta_60')
reg = regime_split(ic_s)
dec = decay_analysis(beta, C)
cov = coverage_stats(beta)
to = turnover_10d(beta)
corrs, max_abs = library_correlation(beta, C, {'DXY': macro_panel('DXY'), 'VIX': macro_panel('VIX')})
print('\n=== cny_beta_60 summary ===')
print('IC10=%.4f ICIR10=%.4f hit=%.3f n=%d' % (m['ic'], m['icir'], m['ic_hit_ratio'], m['n_ic_dates']))
print('regimes:', {k: 'IC=%.4f/ICIR=%.2f/n=%d' % (v['ic'], v['icir'], v['n']) for k, v in reg.items()})
print('decay:', {k: round(v, 4) for k, v in dec.items()})
print('coverage_asset_days=%.3f ge8=%.3f turnover10d=%.3f' % (cov['coverage_asset_days'], cov['coverage_dates_ge8'], to))
print('library_corr:', {k: round(v, 4) for k, v in corrs.items() if np.isfinite(v)})
print('max_abs_library_correlation=%.4f' % max_abs)
gate = abs(m['ic']) >= 0.0070 and abs(m['icir']) >= 0.0840
print('GATE PASS: %s' % gate)

out = {'factor_id': 'cny_beta_60', 'visible_through': VISIBLE_THROUGH,
       'metrics': m, 'regimes': reg, 'decay': dec, 'coverage': cov, 'turnover': to,
       'library_corr': corrs, 'max_abs_library_correlation': max_abs, 'gate': gate,
       'latest_cross_section': beta.iloc[-1].sort_values(ascending=False).round(4).to_dict()}
with open('scripts/miner_1_20301114_cny_beta_deepdive.json', 'w') as f:
    json.dump(out, f, indent=1, default=str)
print('\nSaved scripts/miner_1_20301114_cny_beta_deepdive.json')
