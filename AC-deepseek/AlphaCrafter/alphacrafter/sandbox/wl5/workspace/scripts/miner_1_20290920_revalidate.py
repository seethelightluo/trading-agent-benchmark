# -*- coding: utf-8 -*-
"""miner_1 2029-09-20: periodic re-validation of ALL effective library factors.
Gates (same-horizon admission): |IC| >= 0.0070 and |ICIR| >= 0.0840 at 10d horizon.
Data visible through 2029-09-19 (previous completed trading day).
Recomputes each factor from raw OHLCV; no persistence in this script.
"""
import sys, json, os
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
import miner3_lib as L

C, V, H, Lw, O = L.load_close_panel(4000)
R = C.pct_change()
print('Panel: %s -> %s | %d dates x %d assets' % (C.index.min().date(), C.index.max().date(), len(C), C.shape[1]))

def load_macro(name):
    df = pd.read_csv('../persistent/index_data/%s.csv' % name, parse_dates=['date'])
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df = df.set_index('date').sort_index()
    return df['close'].reindex(C.index).ffill()

DXY = load_macro('DXY'); VIX = load_macro('VIX')
RDXY = DXY.pct_change(); RVIX = VIX.pct_change()
print('Macros through %s' % DXY.index.max().date())

def roll_beta(x, ref, win=60, minp=60):
    cov = x.rolling(win, min_periods=minp).cov(ref)
    var = ref.rolling(win, min_periods=minp).var()
    return cov.div(var, axis=0).replace([np.inf, -np.inf], np.nan)

# ---------------- recompute library factor panels ----------------
factors = {}

# trend_r2_30_signed
lp = np.log(C)
def signed_r2(series):
    y = series.values
    n = len(y)
    out = np.full(n, np.nan)
    x = np.arange(n, dtype=float)
    xm = x.mean(); xv = ((x - xm) ** 2).sum()
    for i in range(n):
        lo = max(0, i - 29)
        seg = y[lo:i + 1]
        if len(seg) < 18 or np.isnan(seg).any():
            continue
        s = seg - seg.mean()
        b = (s * x[lo:i + 1]).sum() / xv
        ss_tot = (s ** 2).sum()
        ss_res = ss_tot - (s * x[lo:i + 1]).sum() ** 2 / xv
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        out[i] = np.sign(b) * max(r2, 0.0)
    return pd.Series(out, index=series.index)
factors['trend_r2_30_signed'] = lp.apply(signed_r2)

# semi_down_ratio_20
dn = np.sqrt((R.where(R < 0, 0.0) ** 2).rolling(20).mean())
up = np.sqrt((R.where(R > 0, 0.0) ** 2).rolling(20).mean())
factors['semi_down_ratio_20'] = dn / up.replace(0, np.nan) - 1.0

# vol_of_vol20x60
factors['vol_of_vol20x60'] = R.rolling(20).std().rolling(60).std()

# mom_120d_skip5 / mom_10d_skip5
factors['mom_120d_skip5'] = C.shift(5) / C.shift(125) - 1.0
factors['mom_10d_skip5'] = C.shift(5) / C.shift(15) - 1.0

# time_under_water_120
factors['time_under_water_120'] = (C.rolling(120, min_periods=60).max() - C) / C.rolling(120, min_periods=60).max()

# vix_beta_cond_60x20
factors['vix_beta_cond_60x20'] = -roll_beta(R, RVIX, 60, 30) * (VIX / VIX.shift(20) - 1.0)

# dxy_beta_60
factors['dxy_beta_60'] = roll_beta(R, RDXY, 60)

# WTI_BETA_60
factors['WTI_BETA_60'] = roll_beta(R, R['WTI'], 60)

# kurt_20
def kurt20(s):
    m2 = s.rolling(20, min_periods=8).mean()
    d2 = (s - m2) ** 2
    d4 = (s - m2) ** 4
    return d4.rolling(20, min_periods=8).mean() / (d2.rolling(20, min_periods=8).mean() ** 2).replace(0, np.nan) - 3.0
factors['kurt_20'] = kurt20(R)

# tail_ratio_20
def tail_ratio(s):
    q95 = s.rolling(20, min_periods=10).quantile(0.95)
    q05 = s.rolling(20, min_periods=10).quantile(0.05)
    return q95 / q05.abs().replace(0, np.nan)
factors['tail_ratio_20'] = tail_ratio(R)

# ---------------- validation ----------------
def recent_summary(s):
    out = {}
    for name, lo in [("2027+", "2027-01-01"), ("2028+", "2028-01-01"), ("2029", "2029-01-01"), ("2029H2", "2029-07-01")]:
        sub = s[s.index >= lo]
        if len(sub) >= 20:
            out[name] = {'ic': round(sub.mean(), 4),
                         'icir': round(sub.mean() / sub.std(), 4) if sub.std() > 0 else 0.0,
                         'n': int(len(sub))}
    return out

print('\n%-24s %8s %8s %6s %5s | %9s %9s %9s | %s' % (
    'factor', 'ic10', 'icir10', 'hit', 'n', 'ic27+', 'ic28+', 'ic29', 'gate'))
results = {}
for fid, panel in factors.items():
    s = L.rank_ic(panel, R.shift(-10))
    if s is None or len(s) < 20:
        print('%-24s ERROR insufficient' % fid); continue
    summ = L.summarize(s, 10, fid)
    summ['regime_recent'] = recent_summary(s)
    rhos, maxrho = L.library_max_rho(panel)
    summ['max_abs_library_correlation'] = maxrho
    results[fid] = summ
    gate = (abs(summ['ic']) >= 0.0070) and (abs(summ['icir']) >= 0.0840)
    r27 = summ['regime_recent'].get('2027+', {}).get('ic', float('nan'))
    r28 = summ['regime_recent'].get('2028+', {}).get('ic', float('nan'))
    r29 = summ['regime_recent'].get('2029', {}).get('ic', float('nan'))
    print('%-24s %8.4f %8.4f %6.3f %5d | %9.4f %9.4f %9.4f | %s' % (
        fid, summ['ic'], summ['icir'], summ['ic_hit_ratio'], summ['n_ic_dates'],
        r27, r28, r29, 'PASS' if gate else 'fail'))

with open('scripts/miner_1_20290920_revalidate_results.json', 'w') as f:
    json.dump({'visible_through': str(C.index.max().date()), 'n_dates': len(C),
               'library_factors': list(factors.keys()), 'results': results}, f, indent=1, default=str)
print('\nSaved scripts/miner_1_20290920_revalidate_results.json')
