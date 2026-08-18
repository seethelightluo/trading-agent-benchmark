# -*- coding: utf-8 -*-
"""miner_3 2030-07-25: explore new candidate factors (batch).
Motivation from recent regime feedback (2030-06/07 blocks):
  - crypto whipsaw (BTC +15.4 after -7.3; ETH 2 down blocks) -> short-horizon reversal / autocorr
  - WTI give-back after +18.6% -> trend stability / acceleration
  - SOX 6th up block in 7, broad risk-on -> cross-asset beta & breadth tilts
  - XAU haven protective -> tail/range structure
Gates: |IC| >= 0.0070 and |ICIR| >= 0.0840 at 10d horizon. Data through 2030-07-24.
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


VIX = load_macro('VIX'); RVIX = VIX.pct_change()

F = {}

# 1) accel_mom_20_60: momentum acceleration normalized by vol
#    (20d ret - 60d ret)/vol20 -> accelerating vs decelerating trends
F['accel_mom_20_60'] = (C.shift(5) / C.shift(25) - 1.0 - (C.shift(5) / C.shift(65) - 1.0)) / R.rolling(20).std().replace(0, np.nan)

# 2) range_ratio_20: (max high - min low)/close over 20d (range expansion)
F['range_ratio_20'] = (H.rolling(20, min_periods=8).max() - Lw.rolling(20, min_periods=8).min()) / C

# 3) updown_asym_5: short-horizon downside/upside semi-vol asymmetry (5d) - whipsaw micro
dn5 = np.sqrt((R.where(R < 0, 0.0) ** 2).rolling(5).mean())
up5 = np.sqrt((R.where(R > 0, 0.0) ** 2).rolling(5).mean())
F['updown_asym_5'] = dn5 / up5.replace(0, np.nan) - 1.0

# 4) ret_autocorr_5: lag-1 autocorr of daily returns over 5d (trend vs mean-revert state)
def autocorr5(s):
    m = s.rolling(5, min_periods=4).mean()
    num = ((s - m) * (s.shift(1) - m.shift(1))).rolling(5, min_periods=4).mean()
    den = (s ** 2).rolling(5, min_periods=4).std().replace(0, np.nan) ** 2
    return (num / den.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
F['ret_autocorr_5'] = autocorr5(R)

# 5) ew_beta_60: beta of each asset to equal-weight cross-asset daily return (60d)
EW = R.mean(axis=1)
cov = R.rolling(60, min_periods=40).cov(EW)
var = EW.rolling(60, min_periods=40).var()
F['ew_beta_60'] = cov.div(var, axis=0).replace([np.inf, -np.inf], np.nan)

# 6) volume_trend_20_60: 20d avg volume / 60d avg volume (volume confirmation)
F['volume_trend_20_60'] = V.rolling(20).mean() / V.rolling(60).mean().replace(0, np.nan)

# 7) rsi_5: 5d RSI (short-horizon mean reversion in whipsaw)
def rsi(s, n=5):
    d = s.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)
F['rsi_5'] = rsi(C)

# 8) gap_ratio_20: 20d mean |open-prev close|/prev close (gap activity)
F['gap_ratio_20'] = (O - C.shift(1)).abs().div(C.shift(1)).rolling(20).mean()

# 9) drawdown_speed_60: 60d drawdown depth per day underwater (recovery pace signal)
dd = (C.rolling(60, min_periods=30).max() - C) / C.rolling(60, min_periods=30).max()
F['dd_speed_60'] = dd / (60.0 / 1.0)

# 10) win_streak_10: fraction of up days in last 10d (breadth of recent move)
F['win_streak_10'] = (R > 0).rolling(10).mean()

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


print('\n%-20s %8s %8s %6s %5s | %8s %8s %8s %8s | %8s | %s' % (
    'candidate', 'ic10', 'icir10', 'hit', 'n', 'ic28+', 'ic29', 'ic29H2', 'ic30', 'maxrho', 'gate'))
results = {}
for fid, panel in F.items():
    s = L.rank_ic(panel, R.shift(-10))
    if s is None or len(s) < 20:
        print('%-20s ERROR insufficient' % fid); continue
    summ = L.summarize(s, 10, fid)
    summ['regime_recent'] = recent_summary(s)
    cov_to = L.coverage_turnover(panel, R, horizon=10)
    summ['coverage_turnover'] = cov_to
    summ['decay'] = L.decay_analysis(panel, R, horizons=(1, 2, 3, 5, 10, 20))
    rhos, maxrho = L.library_max_rho(panel)
    summ['max_abs_library_correlation'] = maxrho
    results[fid] = summ
    gate = (abs(summ['ic']) >= 0.0070) and (abs(summ['icir']) >= 0.0840)
    r28 = summ['regime_recent'].get('2028+', {}).get('ic', float('nan'))
    r29 = summ['regime_recent'].get('2029', {}).get('ic', float('nan'))
    r29h2 = summ['regime_recent'].get('2029H2', {}).get('ic', float('nan'))
    r30 = summ['regime_recent'].get('2030', {}).get('ic', float('nan'))
    print('%-20s %8.4f %8.4f %6.3f %5d | %8.4f %8.4f %8.4f %8.4f | %8.3f | %s' % (
        fid, summ['ic'], summ['icir'], summ['ic_hit_ratio'], summ['n_ic_dates'],
        r28, r29, r29h2, r30, maxrho, 'PASS' if gate else 'fail'))

with open('scripts/miner_3_20300725_explore_results.json', 'w') as f:
    json.dump({'visible_through': str(C.index.max().date()), 'n_dates': len(C),
               'candidates': list(F.keys()), 'results': results}, f, indent=1, default=str)
print('\nSaved scripts/miner_3_20300725_explore_results.json')
