# -*- coding: utf-8 -*-
"""miner_3 2030-08-22: explore new candidate factors (batch).
Motivation from 2030-08-08 block feedback (visible through 2030-08-21):
  - BTC -24% crash vs ETH +21.8% V-rebound -> short-horizon reversal / z-score
  - WTI -17% give-back after +6.4% -> trend stability / vol-adj breakout distance
  - XAU haven bid (+4.8% prior block, +0.7% this) -> haven beta (xau_beta)
  - NDX/SOX selloff, SX5E Europe relief -> rate sensitivity (us10y_beta), carry (usdjpy_beta)
  - Defensive tilt partially protective -> downside beta (asymmetric systematic risk)
Gates: |IC| >= 0.0070 and |ICIR| >= 0.0840 at 10d horizon. Data through 2030-08-21.
"""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
import miner3_lib as L

C, V, H, Lw, O = L.load_close_panel(5000)
R = C.pct_change()
print('Panel: %s -> %s | %d dates x %d assets' % (C.index.min().date(), C.index.max().date(), len(C), C.shape[1]))


def load_macro(name):
    df = pd.read_csv('../persistent/index_data/%s.csv' % name, parse_dates=['date'])
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df = df.set_index('date').sort_index()
    s = df['close'].reindex(C.index).ffill()
    return s[s.index <= C.index.max()]


def rolling_beta(ret_i, ret_m, win, minp):
    cov = ret_i.rolling(win, min_periods=minp).cov(ret_m)
    var = ret_m.rolling(win, min_periods=minp).var()
    return cov.div(var.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


EW = R.mean(axis=1)
XAU_R = R['XAU']
US10Y_R = R['US10Y']
USDJPY = load_macro('USDJPY'); R_USDJPY = USDJPY.pct_change()
VOL20 = R.rolling(20, min_periods=12).std()

F = {}

# 1) zscore_20: (close - SMA20)/vol20 - mean-reversion z-score (whipsaw reversal)
F['zscore_20'] = (C - C.rolling(20, min_periods=12).mean()) / VOL20

# 2) xau_beta_60: beta of each asset to XAU daily returns (haven co-movement)
F['xau_beta_60'] = rolling_beta(R, XAU_R, 60, 40)

# 3) us10y_beta_60: beta to US10Y daily returns (rate sensitivity)
F['us10y_beta_60'] = rolling_beta(R, US10Y_R, 60, 40)

# 4) usdjpy_beta_60: beta to USDJPY daily returns (carry / risk-sentiment)
F['usdjpy_beta_60'] = rolling_beta(R, R_USDJPY, 60, 40)

# 5) downside_beta_60: beta of asset to EW market computed on down-market days only
down_mask = (EW < 0)
R_EW_dn = EW.where(down_mask)
F['downside_beta_60'] = rolling_beta(R, R_EW_dn, 60, 40)

# 6) hi_lo_pos_20: (close - 20d low)/(20d high - 20d low) - stochastic position
F['hi_lo_pos_20'] = (C - Lw.rolling(20, min_periods=12).min()) / (H.rolling(20, min_periods=12).max() - Lw.rolling(20, min_periods=12).min()).replace(0, np.nan)

# 7) breakout_dist_20: (close - 20d max)/vol20 - distance from recent high in vol units
F['breakout_dist_20'] = (C - H.rolling(20, min_periods=12).max()) / VOL20

# 8) up_down_beta_ratio_60: downside beta / upside beta (asymmetric systematic risk)
up_mask = (EW > 0)
R_EW_up = EW.where(up_mask)
db = rolling_beta(R, R_EW_dn, 60, 40)
ub = rolling_beta(R, R_EW_up, 60, 40)
F['up_down_beta_ratio_60'] = db / ub.replace(0, np.nan).abs()

# ---------------- validation ----------------
def recent_summary(s):
    out = {}
    for name, lo in [("2028+", "2028-01-01"), ("2029", "2029-01-01"), ("2029H2", "2029-07-01"),
                     ("2030", "2030-01-01"), ("2030H1", "2030-01-01"), ("2030Q2", "2030-04-01")]:
        sub = s[s.index >= lo]
        if len(sub) >= 15:
            out[name] = {'ic': round(sub.mean(), 4),
                         'icir': round(sub.mean() / sub.std(), 4) if sub.std() > 0 else 0.0,
                         'n': int(len(sub))}
    return out


print('\n%-22s %8s %8s %6s %5s | %8s %8s %8s %8s | %8s | %s' % (
    'candidate', 'ic10', 'icir10', 'hit', 'n', 'ic28+', 'ic29', 'ic29H2', 'ic30', 'maxrho', 'gate'))
results = {}
for fid, panel in F.items():
    s = L.rank_ic(panel, R.shift(-10))
    if s is None or len(s) < 20:
        print('%-22s ERROR insufficient' % fid); continue
    summ = L.summarize(s, 10, fid)
    summ['regime_recent'] = recent_summary(s)
    summ['coverage_turnover'] = L.coverage_turnover(panel, R, horizon=10)
    summ['decay'] = L.decay_analysis(panel, R, horizons=(1, 2, 3, 5, 10, 20))
    rhos, maxrho = L.library_max_rho(panel)
    summ['max_abs_library_correlation'] = maxrho
    results[fid] = summ
    gate = (abs(summ['ic']) >= 0.0070) and (abs(summ['icir']) >= 0.0840)
    r28 = summ['regime_recent'].get('2028+', {}).get('ic', float('nan'))
    r29 = summ['regime_recent'].get('2029', {}).get('ic', float('nan'))
    r29h2 = summ['regime_recent'].get('2029H2', {}).get('ic', float('nan'))
    r30 = summ['regime_recent'].get('2030', {}).get('ic', float('nan'))
    print('%-22s %8.4f %8.4f %6.3f %5d | %8.4f %8.4f %8.4f %8.4f | %8.3f | %s' % (
        fid, summ['ic'], summ['icir'], summ['ic_hit_ratio'], summ['n_ic_dates'],
        r28, r29, r29h2, r30, maxrho, 'PASS' if gate else 'fail'))

with open('scripts/miner_3_20300822_explore_results.json', 'w') as f:
    json.dump({'visible_through': str(C.index.max().date()), 'n_dates': len(C),
               'candidates': list(F.keys()), 'results': results}, f, indent=1, default=str)
print('\nSaved scripts/miner_3_20300822_explore_results.json')
