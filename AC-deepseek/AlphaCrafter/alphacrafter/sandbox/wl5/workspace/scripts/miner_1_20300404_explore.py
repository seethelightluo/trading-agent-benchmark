# -*- coding: utf-8 -*-
"""miner_1 2030-04-04: exploration batch of new factor candidates.

Motivation from recent cycles (memory):
- WTI/BTC momentum whipsaw (raw cross-sectional momentum can't separate them)
- drawdown-gating helped (BTC 5-of-6 down blocks avoided by tuw)
- semi/tech vs energy/crypto alternating dispersion -> consistency/breadth signals
Candidates explored (single idea per construction, all cross-asset 15-instrument):
  A. momentum consistency (up-day ratio signed by direction)
  B. efficiency ratio (Kaufman path efficiency of trend)
  C. range position (close location within 20d high-low band)
  D. drawdown depth 60 (peak-to-trough depth, not duration)
  E. usdjpy_beta_60 (yen-carry proxy, macro observation-only)
  F. eurusd_beta_60 (carry/risk proxy)
  G. rel_mom_20_xs (cross-sectional relative momentum)
  H. down_dev_ratio_60 (return per unit downside deviation)
  I. up_down_vol_asym_60 (upside vol minus downside vol)
Data visible through 2030-04-03. No persistence here; results JSON for review.
"""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from miner_1_20300404_common import (
    TRADABLE, MACRO, VISIBLE_THROUGH, CURRENT_DATE, ohlcv_panels, macro_panel,
    rank_ic_series, summarize_ic, coverage_turnover, regime_breakdown,
)

C, V, H, Lw, O = [ohlcv_panels()[k] for k in ['close', 'volume', 'high', 'low', 'open']]
R = C.pct_change()
print('Panel: %s -> %s | %d dates x %d assets' % (C.index.min().date(), C.index.max().date(), len(C), C.shape[1]))

DXY = macro_panel('DXY'); VIX = macro_panel('VIX'); USDJPY = macro_panel('USDJPY')
EURUSD = macro_panel('EURUSD'); USDCNY = macro_panel('USDCNY')
RDXY = DXY.pct_change(); RVIX = VIX.pct_change(); RJPY = USDJPY.pct_change()
REUR = EURUSD.pct_change(); RCNY = USDCNY.pct_change()
print('Macros through %s' % DXY.index.max().date())


def roll_beta(x, ref, win=60, minp=60):
    cov = x.rolling(win, min_periods=minp).cov(ref)
    var = ref.rolling(win, min_periods=minp).var()
    return cov.div(var, axis=0).replace([np.inf, -np.inf], np.nan)


cands = {}

# A. momentum consistency: 60d signed momentum scaled by fraction of days moving in that direction
mom60 = C.shift(5) / C.shift(65) - 1.0
up_ratio = (R > 0).rolling(60, min_periods=40).mean()
cands['mom_consistency_60'] = mom60 * (up_ratio - 0.5) * 2.0

# B. efficiency ratio (Kaufman): |net move| / sum of |daily moves| over 60d
abs_sum = R.abs().rolling(60, min_periods=40).sum()
cands['efficiency_ratio_60'] = (C - C.shift(60)).abs() / abs_sum.replace(0, np.nan)

# C. range position: (close - min(low,20)) / (max(high,20) - min(low,20)) - 0.5 (centered)
lo20 = Lw.rolling(20, min_periods=15).min()
hi20 = H.rolling(20, min_periods=15).max()
cands['range_pos_20'] = (C - lo20) / (hi20 - lo20).replace(0, np.nan) - 0.5

# D. drawdown depth 60: peak-to-trough depth (0 = no drawdown, positive depth = deeper)
peak60 = C.rolling(60, min_periods=40).max()
cands['drawdown_depth_60'] = (peak60 - C) / peak60  # positive = deeper underwater

# E. USDJPY beta 60 (yen-carry proxy; observation-only macro)
cands['usdjpy_beta_60'] = roll_beta(R, RJPY, 60, 30)

# F. EURUSD beta 60
cands['eurusd_beta_60'] = roll_beta(R, REUR, 60, 30)

# G. cross-sectional relative momentum 20: own 20d return minus cross-sectional mean
xs_mean = R.rolling(20).mean().mean(axis=1)
cands['rel_mom_20_xs'] = R.rolling(20).mean().sub(xs_mean, axis=0)

# H. down_dev_ratio_60: 60d return / 60d downside deviation
ddn = np.sqrt((R.where(R < 0, 0.0) ** 2).rolling(60, min_periods=40).mean())
cands['down_dev_ratio_60'] = (C.shift(5) / C.shift(65) - 1.0) / ddn.replace(0, np.nan)

# I. up_down_vol_asym_60: upside vol - downside vol (normalized)
dup = np.sqrt((R.where(R > 0, 0.0) ** 2).rolling(60, min_periods=40).mean())
cands['up_down_vol_asym_60'] = (dup - ddn) / (dup + ddn).replace(0, np.nan)

# ---- validation: 10d horizon IC/ICIR + recent regime + coverage/turnover ----
print('\n%-22s %8s %8s %6s %5s | %8s %8s %8s %8s | %6s %6s | %s' % (
    'candidate', 'ic10', 'icir10', 'hit', 'n', 'ic20-26', 'ic27', 'ic28', 'ic29', 'ic30', 'cov', 'gate'))
results = {}
for fid, panel in cands.items():
    s = rank_ic_series(panel, R.shift(-10))
    if s is None or len(s) < 20:
        print('%-22s ERROR insufficient' % fid); continue
    summ = summarize_ic(s, 10, fid)
    summ['regime'] = regime_breakdown(s)
    summ['coverage_turnover'] = coverage_turnover(panel)
    summ['decay_ic_by_horizon'] = {h: round(float(rank_ic_series(panel, R.shift(-h)).mean()), 4)
                                   for h in [1, 2, 3, 5, 10, 20]
                                   if rank_ic_series(panel, R.shift(-h)) is not None
                                   and len(rank_ic_series(panel, R.shift(-h))) >= 20}
    results[fid] = summ
    gate = (abs(summ['ic']) >= 0.0070) and (abs(summ['icir']) >= 0.0840)
    rg = summ['regime'] or {}
    print('%-22s %8.4f %8.4f %6.3f %5d | %8.4f %8.4f %8.4f %8.4f | %6.3f %6.3f | %s' % (
        fid, summ['ic'], summ['icir'], summ['ic_hit_ratio'], summ['n_ic_dates'],
        rg.get('2020', {}).get('ic', float('nan')), rg.get('2027', {}).get('ic', float('nan')),
        rg.get('2028', {}).get('ic', float('nan')), rg.get('2029', {}).get('ic', float('nan')),
        rg.get('2030', {}).get('ic', float('nan')), summ['coverage_turnover']['coverage_asset_days'],
        'PASS' if gate else 'fail'))

with open('scripts/miner_1_20300404_explore_results.json', 'w') as f:
    json.dump({'visible_through': VISIBLE_THROUGH, 'current_date': CURRENT_DATE,
               'n_dates': len(C), 'candidates': results}, f, indent=1, default=str)
print('\nSaved scripts/miner_1_20300404_explore_results.json')
