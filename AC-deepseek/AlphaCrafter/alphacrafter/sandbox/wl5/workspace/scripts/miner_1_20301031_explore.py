# -*- coding: utf-8 -*-
"""miner_1 2030-10-31: exploration batch of new factor candidates.

Motivation from current regime (memory 2030-10-17/2030-10-31, ensemble v17):
- Broad risk-off mid-Oct (ETH -22.6%, WTI -12.5%, SOX -10.5%, SX5E -7.7%, COPPER -10.3%, XAU -6.4%),
  then partial easing end-Oct (NDX +9.31%, COPPER +5.53% rebounds) - violent mean-reversion whipsaw.
- Live 90d cross-sectional IC shows momentum/tail/vol INVERTED (mom_10d -0.253, tail_ratio -0.249,
  vol_of_vol -0.187) -> mean-reversion / rebound / risk-beta factors may now carry signal.
- SOX 6th straight down block; dollar up-cycle (dxy_beta working); crypto whipsaw both ways.
- CN10Y/000300/000688/HSI frozen ballast - China-risk differentiation untested by dxy_beta alone.

Candidates (each a single construction, 15-instrument cross-asset universe):
  A. bollinger_pctb_20   - close position in 20d mean +/- 2*std band (mean reversion)
  B. dd_recovery_20      - 10d bounce strength normalized by 60d drawdown depth (rebound)
  C. vol_adj_rev_10      - negative 10d momentum scaled by 20d vol (vol-scaled reversal)
  D. cny_beta_60         - 60d beta to USDCNY (China-risk exposure; macro observation-only)
  E. vix_up_beta_60      - beta to VIX computed only on VIX-up days (crash sensitivity)
  F. xau_beta_60         - 60d beta to XAU (haven sensitivity)
  G. up_capture_ratio_20 - mean up-day return / mean |down-day| return (capture asymmetry)
  H. vol_shock_10x60     - 10d realized vol / 60d realized vol (vol regime expansion)
Data visible through 2030-10-30. No persistence here; results JSON for review.
"""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from miner_1_20301031_common import (
    WATCH, VISIBLE_THROUGH, CURRENT_DATE, ohlcv_panels, macro_panel,
    rank_ic_series, summarize_ic, decay_analysis, turnover_10d, coverage_stats,
    regime_split, roll_beta, library_correlation,
)

P = ohlcv_panels()
C, H, Lw, O, V = P['close'], P['high'], P['low'], P['open'], P['volume']
R = C.pct_change()
print('Panel: %s -> %s | %d dates x %d assets' % (C.index.min().date(), C.index.max().date(), len(C), C.shape[1]))

DXY = macro_panel('DXY'); VIX = macro_panel('VIX'); USDCNY = macro_panel('USDCNY')
print('Macros through %s' % DXY.index.max().date())

# quick regime snapshot
last = C.index[-1]
print('\nRegime snapshot (10/20/60d returns as of %s):' % last.date())
snap = pd.DataFrame({
    'r10': C.iloc[-1] / C.iloc[-11] - 1,
    'r20': C.iloc[-1] / C.iloc[-21] - 1,
    'r60': C.iloc[-1] / C.iloc[-61] - 1,
}).round(4)
print(snap.sort_values('r10').to_string())
print('DXY 20d: %+.4f | VIX 20d: %+.4f | USDCNY 20d: %+.4f' % (
    DXY.iloc[-1] / DXY.iloc[-21] - 1, VIX.iloc[-1] / VIX.iloc[-21] - 1,
    USDCNY.iloc[-1] / USDCNY.iloc[-21] - 1))

cands = {}

# A. Bollinger %B: (close - ma20) / (2*std20); positive = overbought
ma20 = C.rolling(20, min_periods=15).mean()
sd20 = C.rolling(20, min_periods=15).std()
cands['bollinger_pctb_20'] = (C - ma20) / (2.0 * sd20).replace(0, np.nan)

# B. dd_recovery_20: 10d return / (1 - close/60d peak) -> bounce strength per unit drawdown depth
peak60 = C.rolling(60, min_periods=40).max()
dd_depth = 1.0 - C / peak60
cands['dd_recovery_20'] = (C / C.shift(10) - 1.0) / dd_depth.replace(0, np.nan)

# C. vol_adj_rev_10: -10d momentum / 20d realized vol (reversal scaled by vol)
mom10 = C.shift(5) / C.shift(15) - 1.0
vol20 = R.rolling(20, min_periods=15).std()
cands['vol_adj_rev_10'] = -mom10 / vol20.replace(0, np.nan)

# D. cny_beta_60: beta of asset returns to USDCNY returns (China risk)
RCNY = USDCNY.pct_change()
cands['cny_beta_60'] = roll_beta(R, RCNY, 60, 30)

# E. vix_up_beta_60: beta to VIX on days VIX rose (conditional crash sensitivity)
RVIX = VIX.pct_change()
vix_up = RVIX.where(RVIX > 0)
cov_up = R.rolling(60, min_periods=30).cov(vix_up)
var_up = vix_up.rolling(60, min_periods=30).var()
cands['vix_up_beta_60'] = cov_up.div(var_up, axis=0).replace([np.inf, -np.inf], np.nan)

# F. xau_beta_60: beta of asset returns to XAU returns (haven sensitivity)
RXAU = C['XAU'].pct_change()
cands['xau_beta_60'] = roll_beta(R, RXAU, 60, 30)

# G. up_capture_ratio_20: mean up-day return / mean |down-day return| over 20d
up = R.clip(lower=0)
dn = R.clip(upper=0)
up_mean = up.rolling(20, min_periods=10).mean()
dn_mean = dn.rolling(20, min_periods=10).mean().abs()
cands['up_capture_ratio_20'] = up_mean / dn_mean.replace(0, np.nan)

# H. vol_shock_10x60: 10d realized vol / 60d realized vol (recent vol expansion)
vol10 = R.rolling(10, min_periods=8).std()
vol60 = R.rolling(60, min_periods=40).std()
cands['vol_shock_10x60'] = vol10 / vol60.replace(0, np.nan)

# ---- validation ----
fwd10 = C.shift(-10) / C - 1.0
print('\n%-22s %8s %8s %6s %5s | %8s %8s %8s %8s | %6s %6s | %s' % (
    'candidate', 'ic10', 'icir10', 'hit', 'n', 'ic20-26', 'ic27', 'ic28', 'ic29+', 'cov', 'turn', 'gate'))
results = {}
for fid, panel in cands.items():
    s = rank_ic_series(panel, fwd10)
    if len(s) < 20:
        print('%-22s ERROR insufficient (%d dates)' % (fid, len(s)))
        continue
    summ = summarize_ic(s, fid)
    reg = regime_split(s)
    dec = decay_analysis(panel, C)
    cov = coverage_stats(panel)
    to = turnover_10d(panel)
    corrs, max_abs = library_correlation(panel, C, {'DXY': DXY, 'VIX': VIX})
    summ['regime'] = reg
    summ['decay_ic_by_horizon'] = dec
    summ['coverage'] = cov
    summ['turnover_10d'] = to
    summ['library_corr'] = {k: round(v, 4) for k, v in corrs.items() if np.isfinite(v)}
    summ['max_abs_library_correlation'] = round(max_abs, 4)
    results[fid] = summ
    gate = (abs(summ['ic']) >= 0.0070) and (abs(summ['icir']) >= 0.0840)
    rg = reg
    print('%-22s %8.4f %8.4f %6.3f %5d | %8.4f %8.4f %8.4f %8.4f | %6.3f %6.3f | %s' % (
        fid, summ['ic'], summ['icir'], summ['ic_hit_ratio'], summ['n_ic_dates'],
        rg.get('2020-2022', {}).get('ic', float('nan')), rg.get('2027+', {}).get('ic', float('nan')),
        rg.get('2028+', {}).get('ic', float('nan')), rg.get('2029+', {}).get('ic', float('nan')),
        cov['coverage_asset_days'], to,
        'PASS' if gate else 'fail'))
    print('    decay:', {k: round(v, 4) for k, v in dec.items()},
          '| 90d ic:', round(reg.get('last90d', {}).get('ic', float('nan')), 4),
          '| max_lib_corr:', round(max_abs, 4))

with open('scripts/miner_1_20301031_explore_results.json', 'w') as f:
    json.dump({'visible_through': VISIBLE_THROUGH, 'current_date': CURRENT_DATE,
               'n_dates': len(C), 'n_assets': C.shape[1], 'candidates': results},
              f, indent=1, default=str)
print('\nSaved scripts/miner_1_20301031_explore_results.json')
